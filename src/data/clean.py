import re
import unicodedata
from collections import Counter

import ftfy
from langdetect import detect, LangDetectException, DetectorFactory
from tqdm.auto import tqdm

from configs.config import MAX_WORDS, MIN_WORDS

DetectorFactory.seed = 0


def clean_html_artifacts(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#?\w+;", " ", text)
    return text


def detect_language(text):
    try:
        return detect(text[:1000])
    except LangDetectException:
        return "unknown"


def filter_by_language(documents, lang="en"):
    lang_counts = Counter()
    filtered = []
    for doc in tqdm(documents, desc="Language filtering"):
        detected = detect_language(doc)
        lang_counts[detected] += 1
        if detected == lang:
            filtered.append(doc)
    return filtered, lang_counts


def normalize_text(text):
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.splitlines()
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)
    return text


def remove_empty_lines(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def split_long_document(text, max_words=MAX_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    paragraphs = text.split("\n")
    current_chunk = []
    current_len = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current_len + para_words > max_words and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(para)
        current_len += para_words
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    final_chunks = []
    for chunk in chunks:
        w = chunk.split()
        if len(w) <= max_words:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(w), max_words):
                piece = " ".join(w[i : i + max_words])
                final_chunks.append(piece)
    return [c for c in final_chunks if len(c.split()) >= MIN_WORDS]


def clean_wikitext(text):
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    return text


def clean_documents(documents):
    docs = [clean_html_artifacts(doc) for doc in tqdm(documents, desc="Cleaning HTML")]
    return docs


def normalize_documents(documents):
    return [normalize_text(doc) for doc in tqdm(documents, desc="Normalizing")]


def remove_short_documents(documents, min_words=MIN_WORDS):
    docs = [remove_empty_lines(doc) for doc in documents]
    return [doc for doc in docs if len(doc.split()) >= min_words]


def split_documents(documents, max_words=MAX_WORDS):
    result = []
    for doc in documents:
        result.extend(split_long_document(doc, max_words))
    return result
