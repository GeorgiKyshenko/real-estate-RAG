from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.retriever_node import RetrieverNode
from src.nodes.react_agent import AdvanceSearchAgent


class GraphBuilder:
    def __init__(self, retriever, llm):
        self.retriever_node = RetrieverNode(retriever, llm)
        self.agent = AdvanceSearchAgent(retriever, llm)
        self.graph = None

    def build(self):
        builder = StateGraph(RAGState)
        builder.add_node("retriever", self.retriever_node.retrieve_docs)
        builder.add_node("responder", self.retriever_node.generate_answer)
        builder.add_node("evaluator", self.retriever_node.evaluate_documents)
        builder.add_node("agent", self.agent.agent_answer)

        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "evaluator")

        def decide_route(state: RAGState):
            if state.is_relevant == "yes":
                return "responder"
            else:
                return "agent"

        builder.add_conditional_edges(
            "evaluator",
            decide_route,
            {
                "responder": "responder",
                "agent": "agent"
            }
        )

        builder.add_edge("responder", END)
        builder.add_edge("agent", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
