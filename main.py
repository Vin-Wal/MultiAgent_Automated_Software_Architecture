"""
main.py — LangGraph Pipeline
Wires all 5 agents into a sequential StateGraph.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
import requirements_agent
import architecture_agent
import data_modeler_agent
import critic_agent
import diagram_agent


class ArchState(TypedDict):
    user_input:   str
    requirements: str
    architecture: str
    data_model:   str
    critique:     str
    diagrams:     str


def requirements_node(state):
    return {"requirements": requirements_agent.run(state["user_input"])}

def architecture_node(state):
    return {"architecture": architecture_agent.run(state["requirements"])}

def data_modeler_node(state):
    return {"data_model": data_modeler_agent.run(state["architecture"])}

def critic_node(state):
    return {"critique": critic_agent.run(state["architecture"], state["data_model"])}

def diagram_node(state):
    return {"diagrams": diagram_agent.run(state["architecture"])}


def build_pipeline():
    graph = StateGraph(ArchState)
    graph.add_node("requirements", requirements_node)
    graph.add_node("architecture", architecture_node)
    graph.add_node("data_modeler", data_modeler_node)
    graph.add_node("critic",       critic_node)
    graph.add_node("diagram",      diagram_node)
    graph.set_entry_point("requirements")
    graph.add_edge("requirements", "architecture")
    graph.add_edge("architecture", "data_modeler")
    graph.add_edge("data_modeler", "critic")
    graph.add_edge("critic",       "diagram")
    graph.add_edge("diagram",      END)
    return graph.compile()


def run_pipeline(user_input: str) -> dict:
    app = build_pipeline()
    return app.invoke({
        "user_input":   user_input,
        "requirements": "",
        "architecture": "",
        "data_model":   "",
        "critique":     "",
        "diagrams":     "",
    })


if __name__ == "__main__":
    user_input = input("Describe your system: ").strip()
    result = run_pipeline(user_input)
    for key in ["requirements", "architecture", "data_model", "critique", "diagrams"]:
        print(f"\n{'='*60}\n{key.upper()}\n{'='*60}")
        print(result[key])
