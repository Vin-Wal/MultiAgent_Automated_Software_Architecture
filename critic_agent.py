"""
Critic Agent
Retrieves from: critic_agent corpus
Input:  architecture + data_model (outputs of previous agents)
Output: security review covering OWASP Top 10, NIST CSF 2.0, and risk assessment
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from build_vector_store import load_vector_store, get_agent_retriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vs = load_vector_store()
retriever = get_agent_retriever(vs, "critic_agent", k=5)


def run(architecture: str, data_model: str) -> str:
    docs = retriever.invoke("security vulnerabilities risk assessment OWASP NIST")
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a senior security architect.
Using NIST CSF 2.0, OWASP Top 10, and risk frameworks from the context, review the design.

=== CONTEXT (NIST CSF 2.0, OWASP Top 10, SP800-30, Threat Modeling) ===
{context}

=== ARCHITECTURE + DATA MODEL ===
{architecture}

{data_model}

Produce an Architecture Review with:
1. OWASP TOP 10 ANALYSIS (check each vulnerability, flag risks, mitigations)
2. NIST CSF 2.0 COMPLIANCE (GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, RECOVER)
3. RISK ASSESSMENT (top 5 risks with likelihood and impact scores)
4. ARCHITECTURE WEAKNESSES (single points of failure, bottlenecks, missing components)
5. RECOMMENDATIONS (Critical / High / Medium / Low priority fixes)"""

    return llm.invoke(prompt).content
