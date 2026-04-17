"""
Diagram Agent
Retrieves from: diagram_agent corpus
Input:  architecture (output of architecture_agent)
Output: three PlantUML diagrams — C4 Context, C4 Container, Sequence
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from build_vector_store import load_vector_store, get_agent_retriever
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vs = load_vector_store()
retriever = get_agent_retriever(vs, "diagram_agent", k=5)


def run(architecture: str) -> str:
    docs = retriever.invoke("PlantUML C4 model sequence diagram component")
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a software documentation expert specializing in architecture diagrams.
Using PlantUML syntax from the context, generate diagrams for the architecture below.

=== CONTEXT (PlantUML reference, C4 model, arc42, UML reference) ===
{context}

=== ARCHITECTURE ===
{architecture}

Generate THREE PlantUML diagrams:
1. C4 CONTEXT DIAGRAM (system + external actors/systems)
2. C4 CONTAINER DIAGRAM (internal services, tech choices, communication)
3. SEQUENCE DIAGRAM (most important user flow end-to-end)

Format each as:
```plantuml
@startuml [name]
[valid PlantUML code]
@enduml
```"""

    return llm.invoke(prompt).content
