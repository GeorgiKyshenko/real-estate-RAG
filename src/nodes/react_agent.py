from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state.rag_state import RAGState
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


class AdvanceSearchAgent:
    """Reactive Agent for extensive search."""

    def __init__(self, retriever, llm):
        self.retriever: VectorStoreRetriever = retriever
        self.llm: ChatGoogleGenerativeAI = llm
        self._agent = None

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tavily_tool(self):
        print("Building tools")
        tavily_tool = TavilySearch(
            max_results=3,
            search_depth="advanced",
            include_domains=["imot.bg"],
            description=(
                "Search for properties for sale in Bulgaria on imot.bg. "
                "Use this ONLY when the local documents do not have enough information."
            )
        )
        return tavily_tool

    def _build_agent(self):
        print("Building agent")

        tavily_tool = self._build_tavily_tool()
        tools = [tavily_tool]

        system_prompt = """You are an AI real estate research agent specializing in properties for sale in Bulgaria.
        - Core Responsibilities:
            1.Speak with clarity, formally, with calm tone. Highlight selling points: location advantages, amenities, benefits, unique features.
            2.Provide Structured Property Output
            3.Multiple Results Handling
            4.If the question is unrelated to property listings or auction data in the knowledge base, politely state that you can only assist with available property and auction information.
            5. DON'T tell the user what you arent allowed to do If you dont have the requested information.
        Return only the final useful answer
            """

        self._agent = create_agent(
            self.llm, tools=tools, system_prompt=system_prompt)

    def agent_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]})

        messages = result.get('messages', [])
        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            content = getattr(answer_msg, "content", None)

            if isinstance(content, list):
                answer = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                answer = content

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )
