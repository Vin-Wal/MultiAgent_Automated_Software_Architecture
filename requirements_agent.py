"""
Requirements Agent
Retrieves from: requirements_agent corpus
Input:  user_input (natural language description)
Output: structured SRS with functional + non-functional requirements
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from build_vector_store import load_vector_store, get_agent_retriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vs = load_vector_store()
retriever = get_agent_retriever(vs, "requirements_agent", k=5)


def run(user_input: str) -> str:
    docs = retriever.invoke(user_input)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a senior requirements engineer.
Using the standards and examples below, extract and structure all requirements.

=== CONTEXT (IEEE 29148, EARS, ISO 25010, SRS examples) ===
{context}

=== USER INPUT ===
{user_input}

Produce a complete SRS with:
1. PROJECT OVERVIEW (purpose, scope, users)
2. FUNCTIONAL REQUIREMENTS (EARS syntax, numbered FR-001, FR-002...)
3. NON-FUNCTIONAL REQUIREMENTS (ISO 25010 characteristics, numbered NFR-001...)
4. EXTERNAL INTERFACES (APIs, integrations)
5. CONSTRAINTS & ASSUMPTIONS"""

    return llm.invoke(prompt).content
