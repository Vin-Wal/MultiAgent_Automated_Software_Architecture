"""
Pre-processes raw source documents into clean markdown files ready for indexing.

Handles:
  requirements_docs/
    IEEE830.pdf                                  → docling (with PyMuPDF fallback)
    IEEE 29148-2018 Standard ....html            → BeautifulSoup + markdownify
    srs-template.md                              → copied as-is

  critic_docs/
    (all subdirs contain .md files already)      → no processing needed, indexed directly

Output:
  agent_docs/requirements_docs_processed/        (created by this script)

Usage:
  # Preferred — docling available (Python 3.12 venv):
  /path/to/genai_final_project/.venv/bin/python process_docs.py

  # Fallback — PyMuPDF only (Python 3.13 venv):
  python process_docs.py
"""

import re
import shutil
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

AGENT_DOCS = Path(__file__).parent.parent / "genai_final_project" / "agent_docs"
REQ_SRC    = AGENT_DOCS / "requirements_docs"
REQ_DST    = AGENT_DOCS / "requirements_docs_processed"

# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_") + ".md"


def _clean_md(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── PDF → markdown ─────────────────────────────────────────────────────────────

def _pdf_via_docling(path: Path) -> str:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def _pdf_via_pymupdf(path: Path) -> str:
    """
    Fallback: extract text with font-size-based heading detection.
    Uses PyMuPDF's block structure to infer # / ## headings.
    """
    import fitz
    from collections import Counter

    doc = fitz.open(str(path))

    # Collect all font sizes to find body text baseline
    all_sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        all_sizes.append(round(span["size"], 1))

    if not all_sizes:
        doc.close()
        return ""

    size_counts = Counter(all_sizes)
    body_size   = size_counts.most_common(1)[0][0]
    h2_min      = body_size * 1.15
    h1_min      = body_size * 1.35

    paragraphs: list[str] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            block_parts: list[str] = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    if size >= h1_min:
                        block_parts.append(f"# {text}")
                    elif size >= h2_min:
                        block_parts.append(f"## {text}")
                    else:
                        block_parts.append(text)
            joined = " ".join(block_parts).strip()
            if joined:
                paragraphs.append(joined)

    doc.close()
    return _clean_md("\n\n".join(paragraphs))


def process_pdf(path: Path, dst: Path) -> None:
    out = dst / _slug(path)
    print(f"  [PDF] {path.name}", end=" ", flush=True)

    try:
        md = _pdf_via_docling(path)
        method = "docling"
    except ImportError:
        md = _pdf_via_pymupdf(path)
        method = "pymupdf"
    except Exception as e:
        print(f"\n  docling failed ({e}), falling back to pymupdf ...", end=" ")
        md = _pdf_via_pymupdf(path)
        method = "pymupdf"

    md = _clean_md(md)
    out.write_text(md, encoding="utf-8")
    print(f"→ {out.name}  ({len(md):,} chars, via {method})")


# ── HTML → markdown ────────────────────────────────────────────────────────────

def process_html(path: Path, dst: Path) -> None:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    out = dst / _slug(path)
    print(f"  [HTML] {path.name}", end=" ", flush=True)

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    node = (
        soup.select_one("#content")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )
    raw = markdownify(str(node), heading_style="ATX", bullets="-", strip=["a", "img"])
    md  = _clean_md(raw)
    out.write_text(md, encoding="utf-8")
    print(f"→ {out.name}  ({len(md):,} chars)")


# ── markdown copy ──────────────────────────────────────────────────────────────

def process_md(path: Path, dst: Path) -> None:
    out = dst / _slug(path)
    print(f"  [MD]   {path.name}", end=" ", flush=True)
    shutil.copy2(path, out)
    print(f"→ {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REQ_DST.mkdir(parents=True, exist_ok=True)

    # Check docling availability
    try:
        import docling  # noqa: F401
        print("  docling available — PDF quality will be high.\n")
    except ImportError:
        print("  docling not found — using PyMuPDF fallback for PDFs.\n")
        print("  For best results run with the Python 3.12 venv:")
        print("  /path/to/genai_final_project/.venv/bin/python process_docs.py\n")

    print(f"{'━'*55}")
    print(f"  Processing requirements_docs → requirements_docs_processed")
    print(f"{'━'*55}")

    dispatchers = {
        ".pdf":  process_pdf,
        ".html": process_html,
        ".htm":  process_html,
        ".md":   process_md,
    }

    files = sorted(f for f in REQ_SRC.iterdir() if f.is_file() and not f.name.startswith("."))
    for path in files:
        ext = path.suffix.lower()
        fn  = dispatchers.get(ext)
        if fn:
            fn(path, REQ_DST)
        else:
            print(f"  [SKIP] {path.name} (unsupported type)")

    out_files = list(REQ_DST.glob("*.md"))
    print(f"\n  Done. {len(out_files)} files in {REQ_DST}\n")

    print(f"{'━'*55}")
    print(f"  critic_docs: already markdown — no processing needed.")
    print(f"  {sum(1 for _ in (AGENT_DOCS / 'critic_docs').rglob('*.md'))} .md files ready to index.")
    print(f"{'━'*55}\n")


if __name__ == "__main__":
    main()
