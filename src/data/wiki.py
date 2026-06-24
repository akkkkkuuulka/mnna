from datasets import load_dataset


def load_wikitext_articles():
    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    raw_lines = wiki_dataset["train"]["text"]

    articles = []
    current_article = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("= ") and stripped.endswith(" =") and not stripped.startswith("= ="):
            if current_article:
                text = "\n".join(current_article).strip()
                if len(text) > 50:
                    articles.append(text)
            current_article = []
        else:
            if stripped:
                current_article.append(stripped)
    if current_article:
        text = "\n".join(current_article).strip()
        if len(text) > 50:
            articles.append(text)

    return articles
