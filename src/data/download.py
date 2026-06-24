import gzip
import requests
from tqdm.auto import tqdm


def download_warc(crawl_id, output_path, segment_index=0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"WARC уже скачан: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
        return

    paths_url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/warc.paths.gz"
    print("Скачиваем список WARC-файлов...")
    resp = requests.get(paths_url)
    warc_paths = gzip.decompress(resp.content).decode().strip().split("\n")
    warc_url = f"https://data.commoncrawl.org/{warc_paths[segment_index]}"
    print(f"Скачиваем WARC: {warc_url}")

    with requests.get(warc_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="WARC") as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))
    print("Готово!")
