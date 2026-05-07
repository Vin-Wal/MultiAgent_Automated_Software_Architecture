import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv
from config import GEMINI_MODEL, TEMPERATURE, GOOGLE_API_KEY, RAG_K, invoke_with_retry

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run(user_input: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from build_vector_store import load_vector_store, get_agent_retriever

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    vs        = load_vector_store()
    retriever = get_agent_retriever(vs, "requirements_agent", k=RAG_K)
    docs      = retriever.invoke(user_input)
    context   = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a senior requirements engineer with deep expertise in IEEE 29148,
EARS (Easy Approach to Requirements Syntax), and ISO 25010 quality characteristics.

Using the standards, templates, and examples from the context below, produce a complete
and professional Software Requirements Specification (SRS) for the system described.

=== RAG CONTEXT (IEEE 29148, IEEE 830, EARS syntax, ISO 25010, SRS examples) ===
{context}

=== USER INPUT ===
{user_input}

=== INSTRUCTIONS ===
Think step by step internally but do NOT show your thinking process.
Output ONLY the final SRS document.
Do not include any preamble, introductory text, or step-by-step breakdown.

Begin the document with a title block:
# Software Requirements Specification (SRS)
**Version:** 1.0
**Prepared by:** Requirements Agent

Then produce a complete SRS with the following sections:

## 1. PROJECT OVERVIEW
- 1.1 Purpose: what problem this system solves
- 1.2 Scope: what is included and excluded in this release
- 1.3 Intended Users: list each user type with their primary needs

## 2. FUNCTIONAL REQUIREMENTS
- Use EARS syntax strictly for every requirement:
  "The <system/actor> shall <action> when/while/if <condition>"
  Example: "The <system> shall <action> when <condition>."
  Every FR must follow this exact pattern — no exceptions.
  If a condition is not obvious, use "when the user performs the action" or 
  "when the system processes the request" as a fallback condition.
  No FR should be without a when/while/if clause.
- Number each requirement: FR-001, FR-002, ...
- Each major feature must have its own dedicated subsection
- Do not merge multiple distinct features into one section
- Be specific — avoid vague terms
- Derive implicit requirements the user did not mention but are clearly needed
  including administrative, monitoring, and operational requirements that any
  production system would require

## 3. NON-FUNCTIONAL REQUIREMENTS
- Cover ALL ISO 25010 quality characteristics:
  Performance, Security, Usability, Reliability, Maintainability, Portability, Scalability
- Number each: NFR-001, NFR-002, ...
- Make each measurable with specific metrics (numbers, percentages, time limits)
- Never use vague terms like "minimal", "fast", "easy", "sufficient"
- NFRs must describe system behavior and quality, not implementation details
- Do not include coding standards or development practices as NFRs
- Bad example: "The system shall be easy to use"
- Good example: "The system shall allow a new user to complete core task X within 5 minutes"

## 4. EXTERNAL INTERFACES
- 4.1 User Interfaces — describe each interface at a high level only,
  do not list individual screen elements or UI components
- 4.2 Software Interfaces (APIs, third-party integrations)
- 4.3 Hardware Interfaces — describe any hardware the system depends on
  such as GPS sensors, cameras, or network hardware on user devices.
  Do not write "Not applicable" — identify actual hardware dependencies.

## 5. CONSTRAINTS & ASSUMPTIONS
- 5.1 Constraints (regulatory, technical, budget)
- 5.2 Assumptions"""

    return invoke_with_retry(llm, prompt)


if __name__ == "__main__":
    result = run("Build a ride-sharing app with real-time GPS tracking, payments, and driver ratings")
    print(result)
