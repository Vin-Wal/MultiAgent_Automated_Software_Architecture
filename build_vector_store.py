from pathlib import Path

CORPUS_DIR    = Path("rag_corpus")
VECTOR_DIR    = Path("vector_store")
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

AGENT_TAGS = {
    "requirements_agent": "requirements_agent",
    "architecture_agent": "architecture_agent",
    "data_modeler_agent": "data_modeler_agent",
    "critic_agent":       "critic_agent",
    "diagram_agent":      "diagram_agent",
}

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def load_docs():
    from pypdf import PdfReader
    from langchain_core.documents import Document

    all_docs = []

    for agent_dir in sorted(CORPUS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue

        agent_tag = AGENT_TAGS.get(agent_dir.name, agent_dir.name)
        pdfs = sorted(agent_dir.glob("*.pdf"))
        txts = sorted(agent_dir.glob("*.txt"))

        print(f"\n  {CYAN}[{agent_tag}]{RESET}  {len(pdfs)} PDF(s), {len(txts)} TXT(s)")

        for pdf_path in pdfs:
            try:
                reader = PdfReader(str(pdf_path))
                count = 0
                for i, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue
                    all_docs.append(Document(
                        page_content=text,
                        metadata={
                            "agent":       agent_tag,
                            "source_file": pdf_path.name,
                            "page":        i + 1,
                        },
                    ))
                    count += 1
                print(f"    {GREEN}✓{RESET} {pdf_path.name}  ({count} pages)")
            except Exception as e:
                print(f"    {RED}✗{RESET} {pdf_path.name} — {e}")

        for txt_path in txts:
            if txt_path.name.startswith("."):
                continue
            try:
                text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                all_docs.append(Document(
                    page_content=text,
                    metadata={
                        "agent":       agent_tag,
                        "source_file": txt_path.name,
                        "page":        1,
                    },
                ))
                print(f"    {GREEN}✓{RESET} {txt_path.name}  ({len(text)//1000} KB text)")
            except Exception as e:
                print(f"    {RED}✗{RESET} {txt_path.name} — {e}")

    return all_docs


def chunk_docs(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    # Filter out empty or non-string chunks
    chunks = [c for c in chunks if isinstance(c.page_content, str) and c.page_content.strip()]
    print(f"  Total chunks: {len(chunks)}")
    return chunks


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"  Model : {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_store(chunks, embeddings):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to : {VECTOR_DIR.resolve()}")
    print(f"  Please wait ~3-8 minutes on CPU...")

    # Force all page_content to clean strings
    clean_chunks = []
    for c in chunks:
        text = c.page_content
        if not isinstance(text, str):
            text = str(text)
        # Remove surrogate characters that cause UTF-8 encoding errors
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        text = text.strip()
        if text:
            clean_chunks.append(Document(page_content=text, metadata=c.metadata))

    print(f"  Clean chunks after sanitization: {len(clean_chunks)}")

    return Chroma.from_documents(
        documents=clean_chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def verify_store(vs):
    count = vs._collection.count()
    print(f"\n  {GREEN}✓ {count} vectors indexed{RESET}")
    print(f"\n  Vectors per agent:")
    for tag in AGENT_TAGS.values():
        try:
            n = len(vs._collection.get(where={"agent": tag}, limit=99999)["ids"])
            print(f"    {tag:<25} {n}")
        except Exception:
            pass


def load_vector_store():
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=emb)


def get_agent_retriever(vectorstore, agent_tag: str, k: int = 5):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": {"agent": agent_tag}},
    )


if __name__ == "__main__":
    print(f"\n{BOLD}Building RAG vector store...{RESET}")
    print("=" * 50)

    print(f"\n{BOLD}[1/4] Loading PDFs{RESET}")
    docs = load_docs()
    print(f"\n  Total pages loaded: {len(docs)}")

    print(f"\n{BOLD}[2/4] Chunking{RESET}")
    chunks = chunk_docs(docs)

    print(f"\n{BOLD}[3/4] Loading embedding model{RESET}")
    emb = get_embeddings()

    print(f"\n{BOLD}[4/4] Building vector store{RESET}")
    vs = build_store(chunks, emb)
    verify_store(vs)

    print(f"\n{'=' * 50}")
    print(f"{GREEN}All done!{RESET} Next step: python agents.py")
    print("""
In agents.py load the store like this:

  from build_vector_store import load_vector_store, get_agent_retriever

  vs               = load_vector_store()
  req_retriever    = get_agent_retriever(vs, "requirements_agent")
  arch_retriever   = get_agent_retriever(vs, "architecture_agent")
  db_retriever     = get_agent_retriever(vs, "data_modeler_agent")
  critic_retriever = get_agent_retriever(vs, "critic_agent")
  diag_retriever   = get_agent_retriever(vs, "diagram_agent")
""")
