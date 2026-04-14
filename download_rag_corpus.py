"""
RAG Corpus Downloader 

Downloads all 24 documents for 5 agents in one go.
Already-downloaded files are skipped automatically.

"""

import io
import re
import ssl
import time
import zipfile
import urllib.request
from pathlib import Path

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Browser-style headers on every request — prevents 403s
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/pdf,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Permissive SSL context — fixes Italian university cert issues
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode    = ssl.CERT_NONE

# ── Corpus definition ─────────────────────────────────────────────────────────
# (agent_folder, filename, url, size_mb_approx)
# Special entries use type = "zip" or "scrape" instead of normal download
CORPUS = [

    # ── REQUIREMENTS AGENT ───────────────────────────────────────────────────
    ("requirements_agent", "volere_getting_started.pdf",
     "https://www.volere.org/wp-content/uploads/2018/12/VolereGettingStarted.pdf", 0.3),

    ("requirements_agent", "ieee_29148_2011.pdf",
     "https://nirmt.com/storage/uploads/E-BOOK_BE-INDUSTRIAL-AND-SAFETY/"
     "29148-2011%20-%20ISOIECIEEE%20International%20Standard%20-%20Systems%20and%20software"
     "%20engineering%20--%20Life%20cycle%20processes%20--Requirements.pdf", 1.2),

    ("requirements_agent", "reqview_srs_example.pdf",
     "https://www.reqview.com/papers/"
     "ReqView-Example_Software_Requirements_Specification_SRS_Document.pdf", 0.8),

    ("requirements_agent", "ears_requirements_syntax.pdf",
     "https://ccy05327.github.io/SDD/08-PDF/"
     "Easy%20Approach%20to%20Requirements%20Syntax%20(EARS).pdf", 0.4),

    ("requirements_agent", "enabel_srs_example.pdf",
     "https://www.enabel.be/app/uploads/2025/06/"
     "Annex-A-Detailed-Software-Requirements-Specification-SRS.pdf", 0.6),

    ("requirements_agent", "iso25010_quality_slides.pdf",
     "https://cdn.standards.iteh.ai/samples/35733/"
     "2ca18b477b7845a5b8cae39d6de0c098/ISO-IEC-25010-2011.pdf", 1.3),

    ("requirements_agent", "nlp4re_survey.pdf",
     "https://arxiv.org/pdf/2004.01099", 3.5),

    # ── ARCHITECTURE AGENT ───────────────────────────────────────────────────
    ("architecture_agent", "software_architecture_patterns_oreilly.pdf",
     "https://theswissbay.ch/pdf/Books/Computer%20science/O'Reilly/"
     "software-architecture-patterns.pdf", 5.3),

    ("architecture_agent", "architectural_metapatterns.pdf",
     "https://raw.githubusercontent.com/denyspoltorak/publications/refs/heads/main/"
     "ArchitecturalMetapatterns/Architectural%20Metapatterns.pdf", 52.0),

    ("architecture_agent", "microservices_design_patterns_valuelabs.pdf",
     "https://www.valuelabs.com/wp-content/uploads/2023/05/"
     "Microservices-Design-Patterns.pdf", 0.8),

    ("architecture_agent", "microservices_ibm_redbook.pdf",
     "https://www.redbooks.ibm.com/redbooks/pdfs/sg248275.pdf", 8.5),

    ("architecture_agent", "microservices_pattern_language_map.pdf",
     "https://microservices.io/i/MicroservicePatternLanguage.pdf", 0.2),

    ("architecture_agent", "db_system_concepts_er_ch6.pdf",
     "https://www.db-book.com/slides-dir/PDF-dir/ch6.pdf", 1.8),

    # ── DATA MODELER AGENT ───────────────────────────────────────────────────
    ("data_modeler_agent", "database_design_opentextbc.pdf",
     "https://opentextbc.ca/dbdesign01/open/download?type=pdf", 6.2),

    ("data_modeler_agent", "database_modeling_logical_design.pdf",
     "https://eketab2.files.wordpress.com/2007/09/"
     "databasemodelinganddesignlogical-design.pdf", 4.8),

    ("data_modeler_agent", "mongodb_data_modeling_guide.pdf",
     "https://www.mongodb.com/collateral/mongodb-architecture-guide", 2.0),

    # ── CRITIC AGENT ─────────────────────────────────────────────────────────
    ("critic_agent", "nist_csf_2_0.pdf",
     "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf", 1.6),

    ("critic_agent", "nist_sp800_30_risk_assessment.pdf",
     "https://nvlpubs.nist.gov/nistpubs/legacy/sp/"
     "nistspecialpublication800-30r1.pdf", 1.4),

    ("critic_agent", "nist_sp800_154_threat_modeling.pdf",
     "https://csrc.nist.gov/files/pubs/sp/800/154/ipd/docs/"
     "sp800_154_draft.pdf", 0.6),

    ("critic_agent", "owasp_top10_2021.pdf",
     "https://owasp.org/www-pdf-archive/OWASP_Top_10-2021_en.pdf", 3.2),

    # ── DIAGRAM AGENT ────────────────────────────────────────────────────────
    ("diagram_agent", "plantuml_language_reference.pdf",
     "https://pdf.plantuml.net/PlantUML_Language_Reference_Guide_en.pdf", 9.8),

    ("diagram_agent", "uml_diagram_reference.pdf",
     "https://sparxsystems.com/resources/user-guides/16.1/"
     "model-domains/uml-models.pdf", 2.0),

    # arc42 and C4 model are handled separately below (zip + scrape)
]


# ── Download helpers ──────────────────────────────────────────────────────────

def download_pdf(url, dest, label):
    """Download a single PDF with progress bar. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={**HEADERS, "Referer": "/".join(url.split("/")[:3]) + "/"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as resp, \
             open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                buf = resp.read(65536)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = int(downloaded / total * 40)
                    bar = "█" * pct + "░" * (40 - pct)
                    print(f"\r  [{bar}] {downloaded//1024}KB/{total//1024}KB",
                          end="", flush=True)
        size_kb = dest.stat().st_size // 1024
        print(f"\r  {GREEN}✓ {label}  ({size_kb} KB){RESET}" + " " * 20)
        return True
    except Exception as e:
        print(f"\r  {RED}✗ {label} — {e}{RESET}" + " " * 30)
        if dest.exists():
            dest.unlink()
        return False


def fetch_html(url):
    """Fetch a URL and return plain text (HTML stripped)."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        html = r.read().decode("utf-8", errors="ignore")
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>",  "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, rep in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")]:
        text = text.replace(ent, rep)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


# ── Special handlers ──────────────────────────────────────────────────────────

def download_c4_model(dest_dir):
    """Scrape key pages from c4model.com into a single .txt file."""
    dest = dest_dir / "c4_model_reference.txt"
    label = "c4_model_reference.txt"
    if dest.exists():
        print(f"  {YELLOW}↷ already exists — {label}{RESET}")
        return True

    print(f"  ↓ c4_model_reference  (scraping c4model.com)")
    pages = {
        "Introduction":  "https://c4model.com/introduction",
        "Diagrams":      "https://c4model.com/diagrams",
        "Abstractions":  "https://c4model.com/abstractions",
        "Context":       "https://c4model.com/diagrams/system-context",
        "Container":     "https://c4model.com/diagrams/container",
        "Component":     "https://c4model.com/diagrams/component",
        "Deployment":    "https://c4model.com/diagrams/deployment",
    }
    content = "# C4 Model for Visualising Software Architecture\n"
    content += "# Source: c4model.com (Simon Brown) — free to use\n\n"
    ok_count = 0
    for title, url in pages.items():
        try:
            text = fetch_html(url)
            content += f"\n\n## {title}\nSource: {url}\n\n{text[:4000]}\n"
            ok_count += 1
            time.sleep(0.4)
        except Exception as e:
            content += f"\n\n## {title}\n[Failed to fetch: {e}]\n"

    dest.write_text(content, encoding="utf-8")
    size_kb = dest.stat().st_size // 1024
    print(f"  {GREEN}✓ {label}  ({ok_count}/{len(pages)} pages, {size_kb} KB){RESET}")
    return True


def download_arc42(dest_dir):
    """Download arc42 markdown zip from GitHub and extract to .txt."""
    dest = dest_dir / "arc42_template.txt"
    label = "arc42_template.txt"
    if dest.exists():
        print(f"  {YELLOW}↷ already exists — {label}{RESET}")
        return True

    url = ("https://github.com/arc42/arc42-template/raw/master/dist/"
           "arc42-template-EN-withhelp-markdown.zip")
    print(f"  ↓ arc42_template  (zip from github.com/arc42)")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            data  = b""
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                data += chunk
                if total:
                    pct = int(len(data) / total * 40)
                    bar = "█" * pct + "░" * (40 - pct)
                    print(f"\r  [{bar}] {len(data)//1024}KB/{total//1024}KB",
                          end="", flush=True)
        print()

        combined = ("# arc42 Architecture Documentation Template (EN, with help)\n"
                    "# Source: arc42.org — CC BY-SA 4.0\n\n")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            md_files = sorted(f for f in zf.namelist() if f.endswith(".md"))
            for fname in md_files:
                text = zf.read(fname).decode("utf-8", errors="ignore")
                combined += f"\n\n---\n## File: {fname}\n\n{text}"

        dest.write_text(combined, encoding="utf-8")
        size_kb = dest.stat().st_size // 1024
        print(f"  {GREEN}✓ {label}  ({size_kb} KB){RESET}")
        return True
    except Exception as e:
        print(f"\r  {RED}✗ {label} — {e}{RESET}" + " " * 30)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path("rag_corpus")
    ok, fail, skip = [], [], []
    total_mb = 0.0

    print(f"\n{BOLD}RAG Corpus Downloader — STATGR 5293{RESET}")
    print(f"Columbia University Spring 2026")
    print("=" * 55)

    # ── Standard PDF downloads ────────────────────────────────────────────────
    current_agent = None
    for agent, filename, url, size_mb in CORPUS:
        if agent != current_agent:
            current_agent = agent
            print(f"\n{CYAN}{BOLD}[{agent.upper().replace('_', ' ')}]{RESET}")

        dest = base / agent / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        label = filename.replace("_", " ").replace(".pdf", "")

        if dest.exists():
            print(f"  {YELLOW}↷ already exists — {label}{RESET}")
            skip.append(filename)
            continue

        print(f"  ↓ {label}  (~{size_mb} MB)")
        time.sleep(0.3)
        if download_pdf(url, dest, label):
            ok.append(filename)
            total_mb += size_mb
        else:
            fail.append(filename)

    # ── Diagram agent special files ───────────────────────────────────────────
    diag_dir = base / "diagram_agent"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if current_agent != "diagram_agent":
        print(f"\n{CYAN}{BOLD}[DIAGRAM AGENT — SPECIAL SOURCES]{RESET}")

    # C4 model — scraped HTML
    if download_c4_model(diag_dir):
        ok.append("c4_model_reference.txt")
    else:
        fail.append("c4_model_reference.txt")

    # arc42 — extracted from GitHub zip
    if download_arc42(diag_dir):
        ok.append("arc42_template.txt")
    else:
        fail.append("arc42_template.txt")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_files = len(ok) + len(fail) + len(skip)
    print(f"\n{'=' * 55}")
    print(f"{BOLD}Summary{RESET}")
    print(f"  {GREEN}✓ Downloaded : {len(ok)} files{RESET}")
    print(f"  {YELLOW}↷ Skipped    : {len(skip)} files (already exist){RESET}")
    print(f"  {RED}✗ Failed     : {len(fail)} files{RESET}")
    print(f"  Approx disk  : ~{total_mb:.0f} MB raw PDFs")
    print(f"  Total docs   : {total_files} across 5 agents")

    if fail:
        print(f"\n{RED}Failed — open these in your browser and Save As PDF:{RESET}")
        urls = {f: u for _, f, u, _ in CORPUS}
        for f in fail:
            u = urls.get(f, "see corpus list above")
            print(f"  • {f}\n    {u}")
    else:
        print(f"\n{GREEN}All done! Next step:{RESET}")
        print("  python build_vector_store.py")

    # ── Final file listing ────────────────────────────────────────────────────
    print(f"\n{BOLD}Corpus contents:{RESET}")
    for agent_dir in sorted(base.iterdir()):
        if not agent_dir.is_dir():
            continue
        files = sorted(agent_dir.glob("*"))
        agent_name = agent_dir.name.upper().replace("_", " ")
        print(f"  {CYAN}{agent_name}{RESET}  ({len(files)} files)")
        for f in files:
            print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
