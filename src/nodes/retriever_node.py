from src.state.rag_state import RAGState
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state.evaluation_state import EvaluationState


class RetrieverNode:
    def __init__(self, retriever, llm):
        self.retriever: VectorStoreRetriever = retriever
        self.llm: ChatGoogleGenerativeAI = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def evaluate_documents(self, state: RAGState):
        if not state.retrieved_docs:
            return {"is_relevant": "no"}

        doc_txt = "\n\n".join(
            [doc.page_content for doc in state.retrieved_docs])

        prompt = f"Question: {state.question} \n\n Context: {doc_txt}"
        evaluator = self.llm.with_structured_output(EvaluationState)
        response: EvaluationState = evaluator.invoke([
            ("system", "You are a evaluator assessing relevance of a retrieved document to a user question. If the document contains keywords or semantic meaning related to the user question, grade it as relevant. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."),
            ("human", prompt)
        ])

        print(response.relevance)

        return {"is_relevant": response.relevance}

    def generate_answer(self, state: RAGState) -> RAGState:
        context = "\n\n".join(
            [doc.page_content for doc in state.retrieved_docs])
        system_prompt = f"""You work for real estate agency and you are real estate agent. You need to provide information about the properties, your agency is selling,
        based on information you have.
        Your role is to retrieve relevant property information from the knowledge base and present it persuasively, accurately, and confidently to potential buyers.
        Core Responsibilities:
            1.Use Retrieved Context Only
            2.Speak with clarity, formally, with calm tone. Highlight selling points: location advantages, amenities, benefits, unique features.
            3.Structured Property Output
            4.Multiple Results Handling
            5.If the question is unrelated to property listings or auction data in the knowledge base, politely state that you can only assist with available property and auction information.
            6. DON'T tell the user what you arent allowed to do If you dont have the requested information.

        Context:
        {context}

        Question: {state.question}
        """

        response = self.llm.invoke(system_prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content,
        )
