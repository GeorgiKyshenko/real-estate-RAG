from typing import Optional
from warnings import deprecated
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from src.nodes.nodes import RAGNode
from src.state.rag_state import RAGState
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


@deprecated("Attempted to create a search agent using the search tool, but it is not functional yet. It will be developed in the future for now, please use: RAGNode", category=RAGNode)
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
        tavily = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_domains=["imot.bg"]
        )

        @tool
        def tavily_search(query: str) -> str:
            """
            Search for properties for sale in Bulgaria using Tavily.
            Use this only when retrieved documents do not contain enough information.
            Input should be a clear property search query.
            """
            return tavily.invoke(query)

        return tavily_search

    def _build_agent(self):
        print("Building agent")

        tavily_tool = self._build_tavily_tool()
        tools = [tavily_tool]

        system_prompt = """You are an AI real estate research agent specializing in properties for sale in Bulgaria.
        - Tool Usage Rules
            1.You MUST prioritize retrieved documents over web search.
            2.You have access to the following tool:
              - tavily_search — Use this tool to search the web for up-to-date property listings in Bulgaria.
            3.You may call tavily_search ONLY if no relevant property information exists in the retrieved documents
            4.Do NOT use Tavily for general knowledge questions.
            5.You MUST prioritize retrieved documents over web search.
        Return only the final useful answer
            """

        self._agent = create_agent(
            self.llm, tools=tools, system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
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
