from typing import List, Union
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)


class DocumentProcessor:
    """Utilizing document loading and processing"""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_txt(self, path: Union[str, Path]) -> List[Document]:
        """Load documents from .TXT file"""
        loader = DirectoryLoader(
            path=path,
            glob="**/*.TXT",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        return loader.load()

    # def split_documents(self, documents: List[Document]) -> List[Document]:
    #     return self.splitter.split_documents(documents)
