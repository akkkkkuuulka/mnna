import torch
import triton
import triton.language as tl
import math


# ═══════════════════════════════════════════════════════════════════
#  Эталонная реализация attention на PyTorch (для сравнения)
# ═══════════════════════════════════════════════════════════════════

def torch_attention(Q, K, V, causal=True):
    """Наивная реализация attention: O = softmax(Q @ K^T / sqrt(d)) @ V."""
    scale = Q.shape[-1] ** -0.5
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


# ═══════════════════════════════════════════════════════════════════
#  Triton-ядро: forward pass
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, LSE,
    N_CTX,
    sm_scale,
    BLOCK: tl.constexpr, D: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off = pid_bh * N_CTX * D
    lse_off = pid_bh * N_CTX

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)

    # загружаем блок Q [BLOCK, D]
    q = tl.load(Q + off + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)

    # аккумуляторы online-softmax
    m_i = tl.full([BLOCK], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK, D], dtype=tl.float32)

    # для causal пропускаем полностью замаскированные KV-блоки
    hi = (pid_m + 1) * BLOCK if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK):
        offs_n = start_n + tl.arange(0, BLOCK)

        k = tl.load(K + off + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(V + off + offs_n[:, None] * D + offs_d[None, :],
                    mask=offs_n[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # каузальная + граничная маски
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))
        s = tl.where(offs_n[None, :] < N_CTX, s, float('-inf'))

        # online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m_i = m_new

    # нормализация и сохранение
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    tl.store(Out + off + offs_m[:, None] * D + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    tl.store(LSE + lse_off + offs_m, lse, mask=offs_m < N_CTX)


# ═══════════════════════════════════════════════════════════════════
#  Triton-ядра: backward pass (dK/dV и dQ — два отдельных ядра)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, dO, dK, dV, LSE, Delta,
    N_CTX, sm_scale,
    BLOCK: tl.constexpr, D: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off = pid_bh * N_CTX * D
    lse_off = pid_bh * N_CTX

    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)

    kv_idx = off + offs_n[:, None] * D + offs_d[None, :]
    kv_mask = offs_n[:, None] < N_CTX
    k = tl.load(K + kv_idx, mask=kv_mask, other=0.0)
    v = tl.load(V + kv_idx, mask=kv_mask, other=0.0)

    dk = tl.zeros([BLOCK, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK, D], dtype=tl.float32)

    # для causal: начинаем с первого Q-блока, который видит этот KV-блок
    lo = pid_n * BLOCK if IS_CAUSAL else 0

    for start_m in range(lo, N_CTX, BLOCK):
        offs_m = start_m + tl.arange(0, BLOCK)
        qdo_idx = off + offs_m[:, None] * D + offs_d[None, :]
        q_mask = offs_m[:, None] < N_CTX

        q = tl.load(Q + qdo_idx, mask=q_mask, other=0.0)
        do = tl.load(dO + qdo_idx, mask=q_mask, other=0.0)
        l = tl.load(LSE + lse_off + offs_m, mask=offs_m < N_CTX, other=0.0)
        d = tl.load(Delta + lse_off + offs_m, mask=offs_m < N_CTX, other=0.0)

        # пересчитываем S и P из LSE
        s = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))
        s = tl.where(offs_n[None, :] < N_CTX, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        # dV += P^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dS = P * (dO @ V^T - Delta) * scale
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - d[:, None]) * sm_scale).to(q.dtype)

        # dK += dS^T @ Q
        dk += tl.dot(tl.trans(ds), q)

    tl.store(dK + kv_idx, dk.to(dK.dtype.element_ty), mask=kv_mask)
    tl.store(dV + kv_idx, dv.to(dV.dtype.element_ty), mask=kv_mask)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dO, dQ, LSE, Delta,
    N_CTX, sm_scale,
    BLOCK: tl.constexpr, D: tl.constexpr, IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off = pid_bh * N_CTX * D
    lse_off = pid_bh * N_CTX

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)

    q_idx = off + offs_m[:, None] * D + offs_d[None, :]
    q_mask = offs_m[:, None] < N_CTX

    q = tl.load(Q + q_idx, mask=q_mask, other=0.0)
    do = tl.load(dO + q_idx, mask=q_mask, other=0.0)
    l = tl.load(LSE + lse_off + offs_m, mask=offs_m < N_CTX, other=0.0)
    d = tl.load(Delta + lse_off + offs_m, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK, D], dtype=tl.float32)

    hi = (pid_m + 1) * BLOCK if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK):
        offs_n = start_n + tl.arange(0, BLOCK)
        kv_idx = off + offs_n[:, None] * D + offs_d[None, :]
        kv_mask = offs_n[:, None] < N_CTX

        k = tl.load(K + kv_idx, mask=kv_mask, other=0.0)
        v = tl.load(V + kv_idx, mask=kv_mask, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))
        s = tl.where(offs_n[None, :] < N_CTX, s, float('-inf'))

        p = tl.exp(s - l[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - d[:, None]) * sm_scale).to(k.dtype)

        dq += tl.dot(ds, k)

    tl.store(dQ + q_idx, dq.to(dQ.dtype.element_ty), mask=q_mask)


# ═══════════════════════════════════════════════════════════════════
#  Python-обёртки (wrapper functions)
# ═══════════════════════════════════════════════════════════════════

def flash_attn_forward(Q, K, V, causal=True, BLOCK=64):
    """Запускает forward Triton-ядро. Возвращает (O, LSE)."""
    B, H, N, D = Q.shape
    BH = B * H
    q = Q.reshape(BH, N, D).contiguous()
    k = K.reshape(BH, N, D).contiguous()
    v = V.reshape(BH, N, D).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(BH, N, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK), BH)
    _fwd_kernel[grid](
        q, k, v, o, lse,
        N, 1.0 / math.sqrt(D),
        BLOCK=BLOCK, D=D, IS_CAUSAL=causal,
    )
    return o.reshape(B, H, N, D), lse.reshape(B, H, N)


def flash_attn_backward(Q, K, V, O, LSE, dO, causal=True, BLOCK=64):
    """Запускает два backward Triton-ядра. Возвращает (dQ, dK, dV)."""
    B, H, N, D = Q.shape
    BH = B * H
    q = Q.reshape(BH, N, D).contiguous()
    k = K.reshape(BH, N, D).contiguous()
    v = V.reshape(BH, N, D).contiguous()
    o = O.reshape(BH, N, D).contiguous()
    do = dO.reshape(BH, N, D).contiguous()
    lse = LSE.reshape(BH, N).contiguous()

    # Delta_i = rowsum(dO * O) — вспомогательный вектор для softmax backward
    delta = (do.float() * o.float()).sum(dim=-1)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    sm_scale = 1.0 / math.sqrt(D)
    grid = (triton.cdiv(N, BLOCK), BH)

    _bwd_dkdv_kernel[grid](
        q, k, v, do, dk, dv, lse, delta,
        N, sm_scale,
        BLOCK=BLOCK, D=D, IS_CAUSAL=causal,
    )
    _bwd_dq_kernel[grid](
        q, k, v, do, dq, lse, delta,
        N, sm_scale,
        BLOCK=BLOCK, D=D, IS_CAUSAL=causal,
    )

    return (
        dq.reshape(B, H, N, D),
        dk.reshape(B, H, N, D),
        dv.reshape(B, H, N, D),
    )


# ═══════════════════════════════════════════════════════════════════
#  autograd.Function + nn.Module
# ═══════════════════════════════════════════════════════════════════

class FlashAttentionFunc(torch.autograd.Function):
    """Связывает forward/backward Triton-ядра с autograd PyTorch."""

    @staticmethod
    def forward(ctx, Q, K, V, causal):
        O, LSE = flash_attn_forward(Q, K, V, causal)
        ctx.save_for_backward(Q, K, V, O, LSE)
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors
        dQ, dK, dV = flash_attn_backward(Q, K, V, O, LSE, dO, ctx.causal)
        return dQ, dK, dV, None


class FlashAttention(torch.nn.Module):
    """nn.Module обёртка для использования как слой."""

    def __init__(self, causal=True):
        super().__init__()
        self.causal = causal

    def forward(self, Q, K, V):
        return FlashAttentionFunc.apply(Q, K, V, self.causal)
