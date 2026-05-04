from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    node = soup.select_one("#content") or soup.find("main") or soup.find("article") or soup.body
    return markdownify(str(node), heading_style="ATX", bullets="-", strip=["a", "img"])


def load_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        return ""
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(p for p in pages if p.strip())


_LOADERS = {
    ".md":   load_markdown,
    ".txt":  load_markdown,
    ".html": load_html,
    ".htm":  load_html,
    ".pdf":  load_pdf,
}

SUPPORTED_EXTENSIONS = set(_LOADERS)


def load_document(path: Path) -> str | None:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        return None
    try:
        return loader(path)
    except Exception as e:
        print(f"[loaders] {path.name}: {e}")
        return None
