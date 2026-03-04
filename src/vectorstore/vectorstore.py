from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


class VectorStore:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001")
        self.vectorstore = None
        self.retriever = None

    def create_retriever(self, documents: List[Document], k: int = 5):
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        self.retriever = self.vectorstore.as_retriever(k=k)

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError(
                "Vector store is missing. It must be initialize by calling 'create_retriever' function")
        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        try:
            self.retriever.invoke(query)
        except Exception as e:
            raise ValueError(f"Vector store not initialized. {e}")
