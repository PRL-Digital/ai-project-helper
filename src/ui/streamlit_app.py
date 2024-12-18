import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from typing import Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

from ..config import config
from ..logging_utils import logger
from ..file_utils import get_text_files
from ..document_processor import process_single_file, process_documents
from ..file_monitor import FileMonitor
from ..llm_utils import create_embeddings, create_streaming_chain, query_llm_stream

def load_environment():
    """
    Load environment variables from .env file located one directory up from the current file.
    Returns True if successful, False otherwise.
    """
    try:
        current_dir = Path(__file__).parent
        env_path = current_dir.parent.parent / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
            
            required_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                return False
                
            return True
        else:
            logger.error(f"No .env file found at: {env_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        return False


def select_documents_folder() -> Optional[str]:
    """
    Provides a text input for folder path instead of using Tkinter dialog
    """
    st.sidebar.header("📂 Document Folder Selection")
    
    # Use text input for folder path
    folder_path = st.sidebar.text_input(
        "Enter documents folder path:",
        key="folder_path",
        help="Enter the full path to your documents folder"
    )
    
    if folder_path and folder_path != st.session_state.get('previous_folder_path', ''):
        if os.path.isdir(folder_path):
            st.session_state.previous_folder_path = folder_path
            st.session_state.documents_loaded = False  # Reset the loaded flag when folder changes
            st.sidebar.success(f"✅ Valid folder path: {folder_path}")
            return folder_path
        else:
            st.sidebar.error("❌ Invalid folder path. Please enter a valid directory path.")
            return None
    
    return st.session_state.get('previous_folder_path')

def load_text_files_from_folder(selected_folder: str) -> Optional[tuple[FAISS, FileMonitor]]:
    """Load and process text files from a folder"""
    try:
        logger.info(f"Searching for text files in: {selected_folder}")
        text_files = get_text_files(selected_folder)

        if not text_files:
            logger.warning(f"No text files found in {selected_folder}")
            st.warning("No supported text files found in the selected folder.")
            return None

        logger.info(f"Found {len(text_files)} text files")

        file_monitor = FileMonitor()
        file_monitor.initialize_hashes(text_files, selected_folder)

        text_splitter = CharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        documents = []
        for file in text_files:
            documents.extend(process_single_file(file, text_splitter))

        embeddings = create_embeddings()
        vector_store = process_documents(documents, embeddings)

        return vector_store, file_monitor

    except Exception as e:
        logger.error(f"Error loading text files: {str(e)}")
        st.error(f"Error loading text files: {str(e)}")
        return None

def display_chat_history():
    """Display the chat history in the Streamlit interface"""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

def main():
    st.title("Document Chat Assistant")

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'previous_folder_path' not in st.session_state:
        st.session_state.previous_folder_path = None

    if not load_environment():
        st.error("Failed to load required environment variables. Please check your .env file.")
        return

    documents_folder = select_documents_folder()

    if documents_folder:
        # Load documents only if they haven't been loaded yet or if the folder has changed
        if not st.session_state.documents_loaded:
            with st.spinner("Loading documents... This may take a while."):
                result = load_text_files_from_folder(documents_folder)
                if result:
                    st.session_state.vector_store, st.session_state.file_monitor = result
                    st.session_state.chain = create_streaming_chain(
                        st.session_state.vector_store,
                        config.MODEL_NAME,
                        config.MAX_TOKENS
                    )
                    st.session_state.documents_loaded = True
                    st.success("Documents loaded successfully!")

        if hasattr(st.session_state, 'vector_store'):
            # Display chat interface
            display_chat_history()

            # Input for new question
            if question := st.chat_input("Ask a question about your documents:"):
                st.session_state.chat_history.append(HumanMessage(content=question))
                st.chat_message("user").write(question)

                # Stream the response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = []

                    try:
                        for chunk in query_llm_stream(
                            st.session_state.chain,
                            question,
                            st.session_state.chat_history[:-1]
                        ):
                            if "answer" in chunk:
                                full_response.append(chunk["answer"])
                                response_placeholder.markdown(''.join(full_response))
                            elif "error" in chunk:
                                st.error(chunk["error"])
                                break

                        final_response = ''.join(full_response)
                        st.session_state.chat_history.append(AIMessage(content=final_response))

                    except Exception as e:
                        logger.error(f"Error during streaming: {str(e)}")
                        st.error(f"An error occurred while processing your question: {str(e)}")

    else:
        st.warning("Please select a documents folder.")

if __name__ == "__main__":
    main()