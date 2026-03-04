from src.state.rag_state import RAGState
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI


class RAGNode:
    def __init__(self, retriever, llm):
        self.retriever: VectorStoreRetriever = retriever
        self.llm: ChatGoogleGenerativeAI = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(
            [doc.page_content for doc in state.retrieved_docs])
        system_prompt = f"""You are a professional real estate auctioneer and property sales expert.
        Your role is to retrieve relevant property information from the knowledge base and present it persuasively, accurately, and confidently to potential buyers.
        Core Responsibilities:
            1.Use Retrieved Context Only
            2.Speak with clarity, confidence and calm tone. Highlight selling points: location advantages, amenities, benefits, unique features.
            3.Structured Property Output
            4.Multiple Results Handling
            5.If the question is unrelated to property listings or auction data in the knowledge base, politely state that you can only assist with available property and auction information.
        
        Context:
        {context}
        
        Question: {state.question}
        """

        response = self.llm.invoke(system_prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
