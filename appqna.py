import streamlit as st
import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page settings
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖", layout="wide")

# Set up the LLM + retriever pipeline
def create_chat_chain(vector_db):
    prompt_text = """
    You are a helpful assistant. Answer questions only based on the context provided.
    If the answer isn't in the context, just say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    chat_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chat_chain

# Handle PDF upload and vector DB creation
def handle_pdf_upload(file, api_key):
    if not api_key:
        st.error("Google API key is required.")
        return None

    os.environ["GOOGLE_API_KEY"] = api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            st.warning("No readable content found in the PDF.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        collection_name = f"collection-{uuid.uuid4()}"

        vector_db = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings,
            path="./qdrant_db",
            collection_name=collection_name
        )

        st.session_state.collection_name = collection_name
        return vector_db

    finally:
        os.remove(tmp_path)

# Main app
def main():
    st.sidebar.header("🔧 Settings")

    api_key = st.sidebar.text_input("Google API Key", type="password")
    pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

    if st.sidebar.button("Process PDF"):
        if not pdf_file:
            st.warning("Please upload a PDF.")
        elif not api_key:
            st.warning("Enter your Google API key.")
        else:
            with st.spinner("Processing..."):
                db = handle_pdf_upload(pdf_file, api_key)
                if db:
                    st.session_state.chat_chain = create_chat_chain(db)
                    st.session_state.ready = True
                    st.success("PDF processed. You can now start asking questions.")
                else:
                    st.error("Failed to process the document.")

    st.title("📄 TeeCSer eBook Chatbot")
    st.markdown("Upload a PDF, let the system process it, then ask anything from it.")

    if "ready" not in st.session_state:
        st.session_state.ready = False

    # Input prompt (only show last interaction)
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        if not st.session_state.ready:
            st.warning("Please upload and process a PDF first.")
        else:
            # Show user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate and show assistant response
            with st.spinner("Getting answer..."):
                response = st.session_state.chat_chain.invoke(user_input)
                with st.chat_message("assistant"):
                    st.markdown(response)

if __name__ == "__main__":
    main()
