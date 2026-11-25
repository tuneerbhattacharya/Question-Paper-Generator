"""LangGraph wiring: the agentic RAG workflow.

validate_input -> plan -> retrieve -> grade_context --(weak topics)--> retrieve (rewrite + retry)
                                            |
                                            v
                                  generate -> critique --(problems)--> generate (revise)
                                                 |
                                                 v
                                      answer_key -> check_output -> END
"""
from langgraph.graph import END, START, StateGraph

from qpg import config
from qpg.nodes import Nodes, route_after_critique, route_after_grading, route_after_input
from qpg.state import QPState


def build_graph(llm, judge_llm, search):
    n = Nodes(llm, judge_llm, search)
    g = StateGraph(QPState)

    g.add_node("validate_input", n.validate_input)
    g.add_node("plan", n.plan)
    g.add_node("retrieve", n.retrieve)
    g.add_node("grade_context", n.grade_context)
    g.add_node("generate", n.generate)
    g.add_node("critique", n.critique)
    g.add_node("answer_key", n.answer_key)
    g.add_node("check_output", n.check_output)

    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", route_after_input, {"stop": END, "continue": "plan"})
    g.add_conditional_edges("plan", route_after_input, {"stop": END, "continue": "retrieve"})
    g.add_edge("retrieve", "grade_context")
    g.add_conditional_edges("grade_context", route_after_grading,
                            {"retry": "retrieve", "generate": "generate"})
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"revise": "generate", "answer": "answer_key"})
    g.add_edge("answer_key", "check_output")
    g.add_edge("check_output", END)
    return g.compile()


def run(graph, instructions: str, format_text: str, difficulty: str, doc_sample: str) -> dict:
    return graph.invoke(
        {"instructions": instructions, "format_text": format_text, "difficulty": difficulty,
         "doc_sample": doc_sample, "trace": []},
        config={"recursion_limit": config.RECURSION_LIMIT})
