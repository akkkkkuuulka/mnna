import torch
import lightning as L


class GradNormCallback(L.Callback):
    # Логируем глобальную норму градиентов (и по слоям) -> TensorBoard -> ClearML.
    # Резкий рост нормы = нестабильность обучения; для стабилизации есть grad clipping.
    def __init__(self, log_per_layer=True):
        self.log_per_layer = log_per_layer

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        total = 0.0
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            n = p.grad.detach().norm(2)
            total += n.item() ** 2
            if self.log_per_layer:
                pl_module.log(f"grad_norm/{name}", n, on_step=True, on_epoch=False)
        pl_module.log("grad_norm/global", total ** 0.5, on_step=True, prog_bar=True)
