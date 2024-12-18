import os
import hashlib
from uuid import uuid4
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
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
import chardet

@dataclass
class Config:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 0
    SEARCH_K: int = 5
    FETCH_K: int = 20
    MODEL_NAME: str = 'claude-3-5-haiku-latest'
    MAX_TOKENS: int = 4000
    TEXT_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', 
        '.json', '.yaml', '.yml', '.xml', '.csv', '.log',
        '.sh', '.bash', '.sql', '.ini', '.conf'
    })

# Create a global config instance
config = Config()

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
        self.file_hashes.clear()
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.file_hashes[file_path] = calculate_file_hash(file_path)
    
    def check_for_changes(self, root_folder: str) -> tuple[List[str], List[str], List[str]]:
        """Check for file changes"""
        current_files = get_text_files(root_folder)
        current_file_set = set(current_files)
        existing_file_set = set(self.file_hashes.keys())
        
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
        
        for file_path in removed_files:
            self.file_hashes.pop(file_path, None)
        
        return modified_files, new_files, removed_files

def get_text_files(folder_path: str) -> List[str]:
    """Recursively find all text-based files"""
    text_files = []
    try:
        for root, _, files in os.walk(folder_path):
            logger.info(f"Scanning directory: {root}")
            for file in files:
                if any(file.lower().endswith(ext) for ext in config.TEXT_EXTENSIONS):
                    full_path = os.path.abspath(os.path.join(root, file))
                    text_files.append(full_path)
                    logger.info(f"Found file: {file}")
    except Exception as e:
        logger.error(f"Error walking directory {folder_path}: {str(e)}")
        logger.exception(e)  # This will log the full stack trace
    return text_files

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
    root.withdraw()
    folder_selected = filedialog.askdirectory(
        title="Select Folder Containing Text Files"
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

def detect_encoding(file_path: str) -> str:
    """Detect the encoding of a file"""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    except Exception as e:
        logger.warning(f"Error detecting encoding for {file_path}: {str(e)}")
        return 'utf-8'

def process_single_file(file_path: str, text_splitter: CharacterTextSplitter, selected_folder: str) -> List[Document]:
    """Process a single text file and return list of documents"""
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as file:
            content = file.read()
            relative_path = os.path.relpath(file_path, selected_folder)
            texts = text_splitter.split_text(content)
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": relative_path,
                        "full_path": file_path,
                        "folder": os.path.dirname(relative_path),
                        "filename": os.path.basename(relative_path),
                        "file_type": file_extension[1:] if file_extension else "unknown"
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
    
    needs_rebuild = len(removed_files) > 0 or len(modified_files) > 0
    
    all_docs = []
    all_vectors = []
    
    if needs_rebuild:
        for i in range(vector_store.index.ntotal):
            doc_id = vector_store.index_to_docstore_id[i]
            doc = vector_store.docstore._dict[doc_id]
            if doc.metadata["full_path"] not in removed_files and doc.metadata["full_path"] not in modified_files:
                vector = vector_store.index.reconstruct(i)
                all_docs.append(doc)
                all_vectors.append(vector)
    
    files_to_process = modified_files + new_files
    if files_to_process:
        new_documents = []
        for file_path in files_to_process:
            new_documents.extend(process_single_file(file_path, text_splitter, selected_folder))
        
        if new_documents:
            new_vectors = embeddings.embed_documents([doc.page_content for doc in new_documents])
            
            if needs_rebuild:
                all_docs.extend(new_documents)
                all_vectors.extend(new_vectors)
            else:
                vectors_array = np.array(new_vectors)
                if len(vectors_array.shape) == 1:
                    vectors_array = vectors_array.reshape(1, -1)
                
                start_index = vector_store.index.ntotal
                vector_store.index.add(vectors_array)
                
                for i, doc in enumerate(new_documents):
                    doc_id = str(uuid4())
                    vector_store.docstore._dict[doc_id] = doc
                    vector_store.index_to_docstore_id[start_index + i] = doc_id
    
    if needs_rebuild and all_vectors:
        dimension = len(all_vectors[0])
        new_index = faiss.IndexFlatL2(dimension)
        
        vectors_array = np.array(all_vectors)
        if len(vectors_array.shape) == 1:
            vectors_array = vectors_array.reshape(1, -1)
        new_index.add(vectors_array)
        
        new_docstore = InMemoryDocstore({})
        new_index_to_docstore_id = {}
        
        for i, doc in enumerate(all_docs):
            doc_id = str(uuid4())
            new_docstore._dict[doc_id] = doc
            new_index_to_docstore_id[i] = doc_id
        
        vector_store.index = new_index
        vector_store.docstore = new_docstore
        vector_store.index_to_docstore_id = new_index_to_docstore_id

    if files_to_process:
        logger.info(f"Added/Updated {len(files_to_process)} files in vector store")
    if removed_files:
        logger.info(f"Removed {len(removed_files)} files from vector store")
        logger.info("Rebuilt FAISS index to remove deleted vectors")

@st.cache_resource
def load_text_files_from_folder(selected_folder: str) -> Optional[tuple[FAISS, FileMonitor]]:
    try:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not anthropic_api_key or not openai_api_key:
            raise ValueError("API keys are missing from environment variables.")

        logger.info(f"Searching for text files in: {selected_folder}")
        text_files = get_text_files(selected_folder)
        
        if not text_files:
            logger.warning(f"No text files found in {selected_folder}")
            st.warning("No supported text files found in the selected folder.")
            return None
        
        logger.info(f"Found {len(text_files)} text files")
        for file in text_files:
            logger.info(f"Found file: {file}")

        file_monitor = FileMonitor()
        file_monitor.initialize_hashes(text_files, selected_folder)

        text_splitter = CharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,  # Use config instance
            chunk_overlap=config.CHUNK_OVERLAP  # Use config instance
        )

        documents = []
        for file_path in text_files:
            documents.extend(process_single_file(file_path, text_splitter, selected_folder))
            logger.info(f"Loaded file: {os.path.relpath(file_path, selected_folder)}")

        vector_store = process_documents(documents, OpenAIEmbeddings(api_key=openai_api_key))
        return vector_store, file_monitor

    except Exception as e:
        logger.error(f"Error loading text files: {str(e)}")
        st.error(f"Error loading text files: {str(e)}")
        return None

def display_stats(vector_store: FAISS):
    st.sidebar.title("Statistics")
    doc_count = len(vector_store.docstore._dict)
    st.sidebar.metric("Total Documents", doc_count)
    st.sidebar.metric("Embedding Dimension", vector_store.index.d)

def create_streaming_chain(vector_store: FAISS):
    global config  # Add global config reference
    model = ChatAnthropic(
        model=config.MODEL_NAME,  # Use config instance
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=config.MAX_TOKENS,  # Use config instance
        streaming=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.SEARCH_K}  # Use config instance
    )

    prompt = ChatPromptTemplate.from_template("""You are a senior developer. 
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
    st.title("Langchain RAG Chat (Text File Loader)")

    # Add diagnostic logging
    logger.info("Starting application")
    logger.info(f"Config initialized with extensions: {config.TEXT_EXTENSIONS}")

    load_dotenv()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'vector_store' not in st.session_state or 'file_monitor' not in st.session_state:
        with st.spinner('Select Text Files Folder...'):
            selected_folder = select_documents_folder()
            if selected_folder:
                result = load_text_files_from_folder(selected_folder)
                if result:
                    st.session_state.vector_store, st.session_state.file_monitor = result
                    st.session_state.selected_folder = selected_folder
            else:
                st.warning("No folder selected. Please choose a folder.")
                return

    if st.button("Check for File Changes"):
        st.session_state.force_check = True
    
    if 'force_check' not in st.session_state:
        st.session_state.force_check = False

    if st.session_state.get('vector_store') is not None:
        try:
            modified_files, new_files, removed_files = st.session_state.file_monitor.check_for_changes(
                st.session_state.selected_folder
            )
            
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
                        chunk_size=config.CHUNK_SIZE,  # Use config instance
                        chunk_overlap=config.CHUNK_OVERLAP  # Use config instance
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
                
                st.session_state.force_check = False
        
        except Exception as e:
            logger.error(f"Error checking for file changes: {str(e)}")
            st.error(f"Error checking for file changes: {str(e)}")

        display_stats(st.session_state.vector_store)

        if 'chain' not in st.session_state:
            st.session_state.chain = create_streaming_chain(st.session_state.vector_store)

        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            else:
                st.chat_message("assistant").write(message.content)

        query = st.chat_input("Enter your question:")

        if query:
            try:
                st.session_state.chat_history.append(HumanMessage(content=query))
                st.chat_message("user").write(query)

                answer_container = st.chat_message("assistant").empty()
                full_response = []

                for chunk in query_llm_stream(
                    st.session_state.chain,
                    query,
                    st.session_state.chat_history[:-1]
                ):
                    if chunk.get('answer'):
                        full_response.append(chunk['answer'])
                        answer_container.markdown(''.join(full_response))

                ai_response = ''.join(full_response)
                st.session_state.chat_history.append(AIMessage(content=ai_response))

            except Exception as e:
                logger.error(f"Error querying data: {str(e)}")
                st.error(f"Error querying data: {str(e)}")
    else:
        st.warning("No documents loaded. Please select a folder with text files.")

if __name__ == "__main__":
    main()