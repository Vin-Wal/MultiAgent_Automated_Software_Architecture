"""
Architecture Agent
Retrieves from: architecture_agent corpus
Input:  requirements (output of requirements_agent)
Output: architecture design document with pattern selection and component breakdown
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from build_vector_store import load_vector_store, get_agent_retriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vs = load_vector_store()
retriever = get_agent_retriever(vs, "architecture_agent", k=5)


def run(requirements: str) -> str:
    docs = retriever.invoke(requirements[:500])
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a principal software architect.
Using architecture patterns from the context, design a complete architecture.

=== CONTEXT (Architecture patterns, Microservices, IBM Redbook) ===
{context}

=== REQUIREMENTS ===
{requirements}

Produce an Architecture Design Document with:
1. ARCHITECTURE STYLE (pattern chosen, rationale, trade-offs)
2. SYSTEM COMPONENTS (services, responsibilities, tech stack)
3. COMPONENT INTERACTIONS (REST, messaging, gRPC)
4. DATA FLOW (how data moves, caching strategy)
5. DEPLOYMENT ARCHITECTURE (infrastructure, containers)
6. ARCHITECTURE DECISIONS (key ADRs)"""

    return llm.invoke(prompt).content
