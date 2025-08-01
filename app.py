# app.py
import streamlit as st
import os
import asyncio
import nest_asyncio
import logging
import traceback
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

from modules.vector_store import build_vector_store
from modules.retriever_chain import get_answers_as_json
from modules.llm_setup import get_embeddings_model
from modules.session_handler import initialize_session_state

from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index_storage"
if not os.path.exists(FAISS_INDEX_PATH):
    os.makedirs(FAISS_INDEX_PATH)

async def main():
    """
    The main async function that runs the Streamlit UI and handles logic.
    """
    st.set_page_config(page_title="Intelligent Query-Retrieval System", layout="wide")
    st.title("🧠 Intelligent Query-Retrieval System")

    initialize_session_state()

    with st.sidebar:
        st.header("1. Process Document")

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or "your_actual_openai_key_here" in openai_api_key:
            st.error("OpenAI API Key not found or is a placeholder. Please set it in your .env file.")
            st.stop()
        else:
            st.success("✅ OpenAI API Key loaded.")
            os.environ["OPENAI_API_KEY"] = openai_api_key 

        uploaded_file = st.file_uploader(
            "Upload your document (PDF, DOCX)",
            type=["pdf", "docx"]
        )

        if st.button("Process Document"):
            if uploaded_file is None:
                st.warning("Please upload a document first.")
            else:
                index_name = f"{uploaded_file.name}_{uploaded_file.size}"
                local_index_path = os.path.join(FAISS_INDEX_PATH, index_name)

                with st.spinner("Processing document... Please wait."):
                    try:
                        embeddings = get_embeddings_model()
                        if os.path.exists(local_index_path):
                            st.info("Loading existing vector store from disk...")
                            st.session_state.vector_store = FAISS.load_local(
                                local_index_path,
                                embeddings,
                                allow_dangerous_deserialization=True
                            )
                            st.success("Vector store loaded successfully!")
                        else:
                            st.info("No existing store found. Building a new one...")
                            file_bytes = uploaded_file.getvalue()
                            st.session_state.vector_store = await build_vector_store(
                                file_bytes, uploaded_file.name
                            )
                            st.session_state.vector_store.save_local(local_index_path)
                            st.success("Document processed and vector store saved!")

                        st.session_state.doc_processed = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"\u274c Error: {e}")
                        st.text("\ud83d\udcc4 Full traceback:")
                        st.text(traceback.format_exc())

    st.header("2. Ask Questions")

    if not st.session_state.get("doc_processed"):
        st.info("Please upload and process a document to begin.")
    else:
        questions_input = st.text_area(
            "Enter your questions, one per line:",
            height=250,
            placeholder="What is the grace period for premium payment?\nWhat is the waiting period for pre-existing diseases?\nDoes this policy cover maternity expenses?"
        )

        if st.button("Get Answers"):
            if questions_input.strip() and st.session_state.vector_store:
                questions_list = [q.strip() for q in questions_input.split('\n') if q.strip()]

                if questions_list:
                    with st.spinner("Finding answers... This may take a moment."):
                        try:
                            json_response = await get_answers_as_json(
                                questions_list,
                                st.session_state.vector_store
                            )
                            st.session_state.last_response = json_response
                        except Exception as e:
                            st.error(f"\u274c Error generating response: {e}")
                            st.text(traceback.format_exc())
                            logging.error(f"Error getting answers: {e}")
                else:
                    st.warning("Please enter at least one question.")
            else:
                st.warning("Please enter your questions.")

        if st.session_state.get("last_response"):
            st.subheader("JSON Response:")
            st.json(st.session_state.last_response)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.critical(f"Failed to run the Streamlit app: {e}")
        st.error(f"A critical error occurred: {e}")