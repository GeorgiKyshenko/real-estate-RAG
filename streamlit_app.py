from src.graph_builder.graph_builder import GraphBuilder
from src.vectorstore.vectorstore import VectorStore
from src.documents_ingestion.document_processor import DocumentProcessor
import streamlit as st
from pathlib import Path
import sys
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Property Chat",
    page_icon="",
    layout="centered"
)


def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None


@st.cache_resource
def initialize_rag():
    """Initialize the RAG system"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()

        documents = doc_processor.load_from_txt("data")
        vector_store.create_retriever(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None


def main():
    init_session_state()

    st.title("Bulgaria Real Estate Expert")
    st.markdown("Ask me anything about properties for sale in Bulgaria.")

    # System Initialization
    if not st.session_state.initialized:
        with st.spinner("Loading property data..."):
            rag_system = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What type of property are you looking for?"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.rag_system:
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    result = st.session_state.rag_system.run(prompt)
                    full_response = result['answer']
                    st.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })


if __name__ == "__main__":
    main()
