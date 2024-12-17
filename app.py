import os
import hashlib
from uuid import uuid4
from typing import Optional, List, Dict, Set
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
import tkinter as tk
from tkinter import filedialog
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import logging
import time

@dataclass
class Config:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 0
    SEARCH_K: int = 5
    FETCH_K: int = 20
    MODEL_NAME: str = 'claude-3-5-haiku-latest'
    MAX_TOKENS: int = 4000

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class FileMonitor:
    def __init__(self):
        self.file_hashes: Dict[str, str] = {}
        self.watched_folders: Set[str] = set()
    
    def initialize_hashes(self, file_paths: List[str], root_folder: str):
        """Initialize hash values for all files and store root folder"""
        self.watched_folders.add(root_folder)
        # Reset hashes when initializing
        self.file_hashes.clear()
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.file_hashes[file_path] = calculate_file_hash(file_path)
    
    def check_for_changes(self, root_folder: str) -> tuple[List[str], List[str], List[str]]:
        """Check for file changes"""
        # Get current state of files
        current_files = get_all_ts_files(root_folder)
        current_file_set = set(current_files)
        existing_file_set = set(self.file_hashes.keys())
        
        # Find modified, new, and removed files
        modified_files = []
        for file_path in current_files:
            if os.path.exists(file_path):
                try:
                    current_hash = calculate_file_hash(file_path)
                    if file_path in self.file_hashes:
                        if current_hash != self.file_hashes[file_path]:
                            modified_files.append(file_path)
                    self.file_hashes[file_path] = current_hash
                except (IOError, OSError) as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    continue
        
        new_files = list(current_file_set - existing_file_set)
        removed_files = list(existing_file_set - current_file_set)
        
        # Remove hashes for deleted files    
        for file_path in removed_files:
            self.file_hashes.pop(file_path, None)
        
        return modified_files, new_files, removed_files

def get_all_ts_files(folder_path: str) -> List[str]:
    """Recursively find all TypeScript files"""
    ts_files = []
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.ts'):
                    full_path = os.path.abspath(os.path.join(root, file))
                    ts_files.append(full_path)
    except Exception as e:
        logger.error(f"Error walking directory {folder_path}: {str(e)}")
    return ts_files

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of file content"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except (IOError, OSError) as e:
        logger.error(f"Error calculating hash for {file_path}: {str(e)}")
        return ""

def select_documents_folder():
    """Open a file dialog to select the documents folder"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(
        title="Select Folder Containing TypeScript Files"
    )
    return folder_selected

def process_documents(documents: list, embeddings):
    """Create vector store from processed documents"""
    try:
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_single_file(file_path: str, text_splitter: CharacterTextSplitter, selected_folder: str) -> List[Document]:
    """Process a single TypeScript file and return list of documents"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            relative_path = os.path.relpath(file_path, selected_folder)
            texts = text_splitter.split_text(content)
            
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": relative_path,
                        "full_path": file_path,
                        "folder": os.path.dirname(relative_path),
                        "filename": os.path.basename(relative_path)
                    }
                )
                for text in texts
            ]
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def update_vector_store(
    vector_store: FAISS,
    modified_files: List[str],
    new_files: List[str],
    removed_files: List[str],
    text_splitter: CharacterTextSplitter,
    selected_folder: str
):
    """Update vector store with changed files"""
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # If we have files to remove or modify, we need to rebuild the index
    needs_rebuild = len(removed_files) > 0 or len(modified_files) > 0
    
    # Keep track of documents and vectors to rebuild if necessary
    all_docs = []
    all_vectors = []
    
    # First, collect all existing documents we want to keep
    if needs_rebuild:
        for i in range(vector_store.index.ntotal):
            doc_id = vector_store.index_to_docstore_id[i]
            doc = vector_store.docstore._dict[doc_id]
            if doc.metadata["full_path"] not in removed_files and doc.metadata["full_path"] not in modified_files:
                vector = vector_store.index.reconstruct(i)
                all_docs.append(doc)
                all_vectors.append(vector)
    
    # Process modified and new files
    files_to_process = modified_files + new_files
    if files_to_process:
        new_documents = []
        for file_path in files_to_process:
            new_documents.extend(process_single_file(file_path, text_splitter, selected_folder))
        
        if new_documents:
            new_vectors = embeddings.embed_documents([doc.page_content for doc in new_documents])
            
            if needs_rebuild:
                # Add to our rebuild lists
                all_docs.extend(new_documents)
                all_vectors.extend(new_vectors)
            else:
                # Add directly to the index
                vectors_array = np.array(new_vectors)
                if len(vectors_array.shape) == 1:
                    vectors_array = vectors_array.reshape(1, -1)
                
                # Get the starting index for new documents
                start_index = vector_store.index.ntotal
                
                # Add vectors to the index
                vector_store.index.add(vectors_array)
                
                # Update mappings
                for i, doc in enumerate(new_documents):
                    doc_id = str(uuid4())
                    vector_store.docstore._dict[doc_id] = doc
                    vector_store.index_to_docstore_id[start_index + i] = doc_id
    
    # Rebuild the index if necessary
    if needs_rebuild and all_vectors:
        # Create new FAISS index
        dimension = len(all_vectors[0])
        new_index = faiss.IndexFlatL2(dimension)
        
        # Convert vectors to numpy array and add to index
        vectors_array = np.array(all_vectors)
        if len(vectors_array.shape) == 1:
            vectors_array = vectors_array.reshape(1, -1)
        new_index.add(vectors_array)
        
        # Create new docstore
        new_docstore = InMemoryDocstore({})
        new_index_to_docstore_id = {}
        
        # Add all documents and create mappings
        for i, doc in enumerate(all_docs):
            doc_id = str(uuid4())
            new_docstore._dict[doc_id] = doc
            new_index_to_docstore_id[i] = doc_id
        
        # Update vector store with new data structures
        vector_store.index = new_index
        vector_store.docstore = new_docstore
        vector_store.index_to_docstore_id = new_index_to_docstore_id

    # Log changes
    if files_to_process:
        logger.info(f"Added/Updated {len(files_to_process)} files in vector store")
    if removed_files:
        logger.info(f"Removed {len(removed_files)} files from vector store")
        logger.info("Rebuilt FAISS index to remove deleted vectors")

@st.cache_resource
def load_ts_files_from_folder(selected_folder: str) -> Optional[tuple[FAISS, FileMonitor]]:
    try:
        # Validate API keys
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not anthropic_api_key or not openai_api_key:
            raise ValueError("API keys are missing from environment variables.")

        # Find all .ts files recursively in the selected folder
        ts_files = get_all_ts_files(selected_folder)
        
        if not ts_files:
            st.warning("No TypeScript files found in the selected folder.")
            return None

        # Initialize file monitor
        file_monitor = FileMonitor()
        file_monitor.initialize_hashes(ts_files, selected_folder)

        # Initialize text splitter
        text_splitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        # Process all documents initially
        documents = []
        for file_path in ts_files:
            documents.extend(process_single_file(file_path, text_splitter, selected_folder))
            logger.info(f"Loaded file: {os.path.relpath(file_path, selected_folder)}")

        vector_store = process_documents(documents, OpenAIEmbeddings(api_key=openai_api_key))
        return vector_store, file_monitor

    except Exception as e:
        logger.error(f"Error loading TypeScript files: {str(e)}")
        st.error(f"Error loading TypeScript files: {str(e)}")
        return None

def display_stats(vector_store: FAISS):
    st.sidebar.title("Statistics")
    doc_count = len(vector_store.docstore._dict)
    st.sidebar.metric("Total Documents", doc_count)
    st.sidebar.metric("Embedding Dimension", vector_store.index.d)

def create_streaming_chain(vector_store: FAISS):
    model = ChatAnthropic(
        model=Config.MODEL_NAME,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=Config.MAX_TOKENS,
        streaming=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.SEARCH_K}
    )

    prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant. 
    Answer the question based on the provided context and conversation history.

Conversation History:
{chat_history}

Context: {context}

Question: {input}

Answer: Let me help you with that.""")

    document_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, document_chain)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_llm_stream(chain, query: str, chat_history):
    response = chain.stream({
        "input": query,
        "chat_history": chat_history
    })
    for chunk in response:
        if 'answer' in chunk:
            yield {"answer": chunk['answer']}

def main():
    st.title("Langchain RAG Chat (TypeScript File Loader)")

    # Load environment variables
    load_dotenv()

    # Initialize conversation history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize or get vector store and file monitor from session state
    if 'vector_store' not in st.session_state or 'file_monitor' not in st.session_state:
        with st.spinner('Select TypeScript Files Folder...'):
            selected_folder = select_documents_folder()
            if selected_folder:
                result = load_ts_files_from_folder(selected_folder)
                if result:
                    st.session_state.vector_store, st.session_state.file_monitor = result
                    st.session_state.selected_folder = selected_folder
            else:
                st.warning("No folder selected. Please choose a folder.")
                return  # Exit if no folder selected

    # Add a refresh button to force check for changes
    if st.button("Check for File Changes"):
        st.session_state.force_check = True
    
    # Initialize force_check if it doesn't exist
    if 'force_check' not in st.session_state:
        st.session_state.force_check = False

    if st.session_state.get('vector_store') is not None:
        try:
            # Always check for file changes on each Streamlit rerun
            modified_files, new_files, removed_files = st.session_state.file_monitor.check_for_changes(
                st.session_state.selected_folder
            )
            
            # If there are changes or force_check is True, update the vector store
            if any([modified_files, new_files, removed_files]) or st.session_state.force_check:
                changes = []
                if modified_files:
                    changes.append(f"{len(modified_files)} modified files")
                if new_files:
                    changes.append(f"{len(new_files)} new files")
                if removed_files:
                    changes.append(f"{len(removed_files)} removed files")
                
                if changes:
                    st.info(f"Updating vector store with {', '.join(changes)}...")
                    
                    text_splitter = CharacterTextSplitter(
                        chunk_size=Config.CHUNK_SIZE,
                        chunk_overlap=Config.CHUNK_OVERLAP
                    )
                    
                    try:
                        update_vector_store(
                            st.session_state.vector_store,
                            modified_files,
                            new_files,
                            removed_files,
                            text_splitter,
                            st.session_state.selected_folder
                        )
                        st.success("Files updated successfully!")
                    except Exception as e:
                        logger.error(f"Error updating vector store: {str(e)}")
                        st.error(f"Error updating vector store: {str(e)}")
                elif st.session_state.force_check:
                    st.info("No changes detected in files.")
                
                # Reset force_check
                st.session_state.force_check = False
        
        except Exception as e:
            logger.error(f"Error checking for file changes: {str(e)}")
            st.error(f"Error checking for file changes: {str(e)}")

        # Display statistics
        display_stats(st.session_state.vector_store)

        # Initialize or get streaming chain from session state
        if 'chain' not in st.session_state:
            st.session_state.chain = create_streaming_chain(st.session_state.vector_store)

        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            else:
                st.chat_message("assistant").write(message.content)

        # Chat input
        query = st.chat_input("Enter your question:")

        if query:
            try:
                # Add user message to chat history
                st.session_state.chat_history.append(HumanMessage(content=query))
                st.chat_message("user").write(query)

                # Create answer container
                answer_container = st.chat_message("assistant").empty()
                full_response = []

                # Stream the response
                for chunk in query_llm_stream(
                    st.session_state.chain,
                    query,
                    st.session_state.chat_history[:-1]  # Exclude the latest user message
                ):
                    if chunk.get('answer'):
                        full_response.append(chunk['answer'])
                        answer_container.markdown(''.join(full_response))

                # Add AI response to chat history
                ai_response = ''.join(full_response)
                st.session_state.chat_history.append(AIMessage(content=ai_response))

            except Exception as e:
                logger.error(f"Error querying data: {str(e)}")
                st.error(f"Error querying data: {str(e)}")
    else:
        st.warning("No documents loaded. Please select a folder with TypeScript files.")

if __name__ == "__main__":
    main()