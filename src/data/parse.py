import warnings

from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from warcio.archiveiterator import ArchiveIterator

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass


def extract_text_from_html(html_bytes):
    try:
        html = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "header", "nav", "footer", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text if len(text) > 50 else None


def parse_warc(warc_path, limit=None):
    documents = []
    with open(warc_path, "rb") as f:
        for record in tqdm(ArchiveIterator(f), desc="Parsing WARC"):
            if record.rec_type == "response":
                content_type = record.http_headers.get_header("Content-Type") or ""
                if "text/html" not in content_type:
                    continue
                payload = record.content_stream().read()
                text = extract_text_from_html(payload)
                if text:
                    documents.append(text)
                    if limit is not None and len(documents) >= limit:
                        break
    return documents
