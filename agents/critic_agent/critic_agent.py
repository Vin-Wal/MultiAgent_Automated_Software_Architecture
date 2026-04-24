import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from config import GEMINI_MODEL, TEMPERATURE, GOOGLE_API_KEY, RAG_K, invoke_with_retry

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def extract_score(text: str) -> int:
    patterns = [
        r"OVERALL SCORE[:\s]+(\d+)\s*/\s*10",
        r"SCORE[:\s]+(\d+)\s*/\s*10",
        r"(\d+)\s*/\s*10",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return 5


def run(requirements: str, architecture: str, data_model: str) -> tuple:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from build_vector_store import load_vector_store, get_agent_retriever

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
    )

    vs        = load_vector_store()
    retriever = get_agent_retriever(vs, "critic_agent", k=RAG_K)
    docs      = retriever.invoke("security vulnerabilities risk assessment OWASP NIST threat modeling")
    context   = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a senior security architect and principal technical reviewer.
Your job is to critically evaluate the architecture and data model against security
best practices, architectural patterns, and quality standards.

Using NIST CSF 2.0, OWASP Top 10, STRIDE threat modeling, and architecture patterns
from the context below, perform a comprehensive review.

=== RAG CONTEXT (NIST CSF 2.0, OWASP Top 10, SP800-30, Threat Modeling, Security Cheatsheets) ===
{context}

=== REQUIREMENTS ===
{requirements[:1000]}

=== ARCHITECTURE ===
{architecture[:2000]}

=== DATA MODEL ===
{data_model[:2000]}

Produce a comprehensive Architecture Review with:

## 1. OWASP TOP 10 ANALYSIS
For each category: risk level, specific finding, recommended mitigation

## 2. NIST CSF 2.0 COMPLIANCE
Assess each of the 6 functions using this format for each:

**FUNCTION NAME — Assessment Level**
- Strengths: what is already addressed
- Gaps: what is missing
- Recommendations: specific actions to take

## 3. STRIDE THREAT ANALYSIS
Identify threats: Spoofing, Tampering, Repudiation, Information Disclosure, DoS, Elevation of Privilege

## 4. RISK ASSESSMENT (NIST SP 800-30)
Top 5 risks with likelihood (1-5), impact (1-5), risk score, recommended control

## 5. ARCHITECTURE WEAKNESSES
Single points of failure, bottlenecks, missing components, data consistency risks

## 6. RECOMMENDATIONS
- CRITICAL: must fix before deployment
- HIGH: fix in next iteration
- MEDIUM: address in roadmap
- LOW: nice to have

## 7. OVERALL SCORE
OVERALL SCORE: X/10
Where 1-3=major flaws, 4-6=significant gaps, 7-8=good with minor improvements, 9-10=excellent
Justify the score briefly.

Ensure all markdown tables are properly formatted with no merged or missing rows.
Each table row must be on its own line with proper pipe separators."""

    critique = invoke_with_retry(llm, prompt)
    score    = extract_score(critique)
    return critique, score


# if __name__ == "__main__":
#     critique, score = run(
#         "Build a ride-sharing app.",
#         "Microservices with API Gateway, User Service, Ride Service, Payment Service.",
#         "PostgreSQL for transactional data, Redis for caching."
#     )
#     print(critique)
#     print(f"\nExtracted Score: {score}/10")

if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]

    requirements = (root / "test_inputs" / "requirements.txt").read_text(encoding="utf-8")
    architecture = (root / "test_inputs" / "architecture.txt").read_text(encoding="utf-8")
    data_model   = (root / "test_inputs" / "data_model.txt").read_text(encoding="utf-8")

    critique, score = run(
        requirements[:1000],
        architecture[:2000],
        data_model[:2000]
    )
    print(critique)
    print(f"\nExtracted Score: {score}/10")