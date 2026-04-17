"""
Data Modeler Agent
Retrieves from: data_modeler_agent corpus
Input:  architecture (output of architecture_agent)
Output: data model with ER design, schema, and storage recommendations
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from build_vector_store import load_vector_store, get_agent_retriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vs = load_vector_store()
retriever = get_agent_retriever(vs, "data_modeler_agent", k=5)


def run(architecture: str) -> str:
    docs = retriever.invoke(architecture[:500])
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a senior data architect.
Using database design principles from the context, design the complete data model.

=== CONTEXT (DB Design, ER Modeling, MongoDB patterns) ===
{context}

=== ARCHITECTURE ===
{architecture}

Produce a Data Model Document with:
1. STORAGE TECHNOLOGY (SQL vs NoSQL vs hybrid, with justification)
2. ENTITY RELATIONSHIP MODEL (entities, attributes, relationships, keys)
3. SCHEMA DESIGN (tables/collections, indexes, normalization)
4. DATA ACCESS PATTERNS (frequent queries, read/write ratio)
5. DATA MIGRATION & SEEDING (initial data requirements)"""

    return llm.invoke(prompt).content
