import hashlib


def deduplicate(documents):
    seen_hashes = set()
    unique = []
    for doc in documents:
        h = hashlib.sha256(doc.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)
    rho_dup = len(documents) / len(unique) if unique else 0
    return unique, rho_dup
