import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from typing import Optional, AsyncIterator
import asyncio

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

import tiktoken
from openai import OpenAI
import anthropic

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

def update_document_and_cost_tracking(text_files):
    """
    Update session state with document count and cost tracking
    """
    # Document Count Tracking
    st.session_state.document_count = len(text_files)

    # Initialize cost tracking if not exists
    if 'total_embedding_cost' not in st.session_state:
        st.session_state.total_embedding_cost = 0.0
    if 'total_llm_cost' not in st.session_state:
        st.session_state.total_llm_cost = 0.0

def calculate_embedding_cost(documents):
    """
    Calculate the cost of embeddings using OpenAI's pricing
    """
    # OpenAI embedding pricing (as of 2023)
    # $0.0001 / 1K tokens
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(encoding.encode(doc.page_content)) for doc in documents)
        embedding_cost = (total_tokens / 1000) * 0.0001
        return embedding_cost
    except Exception as e:
        logger.error(f"Error calculating embedding cost: {str(e)}")
        return 0.0

class LLMCostTracker:
    """
    Tracks LLM costs including streaming responses with accurate accumulation
    """
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Updated model pricing per 1K tokens
        self.model_prices = {
            'claude': {
                'input': 0.00163,
                'output': 0.00551
            },
            'gpt-4': {
                'input': 0.03,
                'output': 0.06
            },
            'gpt-3.5': {
                'input': 0.0015,
                'output': 0.002
            }
        }
        
        # Keep running totals
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
    
    def get_model_prices(self, model_name: str) -> tuple[float, float]:
        """Get input and output prices for a model with fallback handling"""
        model_key = next((k for k in self.model_prices.keys() if k in model_name.lower()), 'gpt-3.5')
        return self.model_prices[model_key]['input'], self.model_prices[model_key]['output']
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with error handling"""
        try:
            if not text:
                return 0
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Token counting error: {str(e)}")
            return 0
    
    def calculate_prompt_cost(self, messages: list, model_name: str) -> float:
        """Calculate cost of the input prompt"""
        try:
            input_text = ""
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage)):
                    input_text += f"{msg.type}: {msg.content}\n"
                elif isinstance(msg, dict):
                    input_text += f"{msg.get('type', 'unknown')}: {msg.get('content', '')}\n"
                elif isinstance(msg, str):
                    input_text += f"{msg}\n"
            
            tokens = self.count_tokens(input_text)
            self.input_tokens += tokens
            
            input_price_per_1k, _ = self.get_model_prices(model_name)
            cost = (tokens / 1000) * input_price_per_1k
            self.total_cost += cost
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating prompt cost: {str(e)}")
            return 0.0
    
    def calculate_streaming_chunk_cost(self, chunk: str, model_name: str) -> float:
        """Calculate cost of a response chunk"""
        try:
            tokens = self.count_tokens(chunk)
            self.output_tokens += tokens
            
            _, output_price_per_1k = self.get_model_prices(model_name)
            cost = (tokens / 1000) * output_price_per_1k
            self.total_cost += cost
            return cost
            
        except Exception as e:
            logger.error(f"Error calculating chunk cost: {str(e)}")
            return 0.0
    
    def get_total_cost(self) -> float:
        """Get the total accumulated cost"""
        return self.total_cost
    
    def get_token_counts(self) -> tuple[int, int]:
        """Get the total input and output token counts"""
        return self.input_tokens, self.output_tokens

async def consume_stream(async_generator):
    """Helper function to consume an async generator"""
    response = []
    async for chunk in async_generator:
        response.append(chunk)
    return response

async def query_llm_stream(chain, question: str, chat_history: list):
    """
    Query LLM with streaming response and cost tracking
    """
    try:
        cost_tracker = LLMCostTracker()
        total_response = []
        
        # Calculate prompt cost
        prompt_messages = chat_history + [HumanMessage(content=question)]
        prompt_cost = cost_tracker.calculate_prompt_cost(prompt_messages, config.MODEL_NAME)
        
        # Update session state with prompt cost - use atomic update
        if 'total_llm_cost' not in st.session_state:
            st.session_state.total_llm_cost = 0.0
        st.session_state.total_llm_cost = st.session_state.total_llm_cost + prompt_cost
        
        # Stream response and track costs
        formatted_history = "\n".join(f"{msg.type}: {msg.content}" for msg in chat_history)
        
        async for chunk in chain.astream({
            "input": question,
            "chat_history": formatted_history
        }):
            content = None
            if isinstance(chunk, dict):
                content = (chunk.get("answer") or 
                         chunk.get("response") or 
                         chunk.get("output", ""))
            elif isinstance(chunk, str):
                content = chunk
                
            if content:
                # Calculate and update cost for this chunk
                chunk_cost = cost_tracker.calculate_streaming_chunk_cost(content, config.MODEL_NAME)
                st.session_state.total_llm_cost = st.session_state.total_llm_cost + chunk_cost
                
                # Store chunk for full response tracking
                total_response.append(content)
                yield content
        
        # Log final costs for debugging
        logger.debug(f"""
            Final cost breakdown:
            Prompt cost: ${prompt_cost:.4f}
            Response cost: ${st.session_state.total_llm_cost - prompt_cost:.4f}
            Total cost: ${st.session_state.total_llm_cost:.4f}
            Total response length: {len(''.join(total_response))}
        """)

    except Exception as e:
        logger.error(f"Error in streaming query: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        yield f"Error: {str(e)}"

def display_sidebar_info():
    """
    Initial sidebar setup with headers and metrics
    """
    with st.sidebar:
        st.header("📊 Document Insights")

        # Document Count
        st.metric("Total Documents", 
            st.session_state.get('document_count', 0)
        )

        # Cost Tracking section
        st.subheader("Cost Tracking")
        update_sidebar_metrics()

# Modify the load_text_files_from_folder function
def load_text_files_from_folder(selected_folder: str) -> Optional[tuple[FAISS, FileMonitor]]:
    try:
        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        text_files = get_text_files(selected_folder)

        if not text_files:
            return None

        # Create file monitor
        file_monitor = FileMonitor(selected_folder)

        # Update document count
        update_document_and_cost_tracking(text_files)

        # Process documents
        documents = []
        for file in text_files:
            documents.extend(process_single_file(file, text_splitter))

        # Calculate embedding cost
        embedding_cost = calculate_embedding_cost(documents)
        st.session_state.total_embedding_cost += embedding_cost

        embeddings = create_embeddings()
        vector_store = process_documents(documents, embeddings)

        return vector_store, file_monitor

    except Exception as e:
        logger.error(f"Error loading text files: {str(e)}")
        return None

def display_chat_history():
    """Display the chat history in the Streamlit interface"""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

async def stream_chat_response(chain, question: str, chat_history: list) -> AsyncIterator[str]:
    """
    Unified streaming response handler that works with both async and sync chains
    """
    try:
        formatted_history = "\n".join(f"{msg.type}: {msg.content}" for msg in chat_history)
        
        if hasattr(chain, 'astream'):
            async for chunk in chain.astream({
                "input": question,
                "chat_history": formatted_history
            }):
                if isinstance(chunk, dict):
                    content = (chunk.get("answer") or 
                             chunk.get("response") or 
                             chunk.get("output", ""))
                    if content:
                        yield content
                elif isinstance(chunk, str):
                    yield chunk
        else:
            # Fallback for sync chains
            response = chain({
                "input": question,
                "chat_history": formatted_history
            })
            yield response.get("answer", "") or response.get("response", "") or response.get("output", "")
            
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"Error generating response: {str(e)}"

async def handle_chat_interaction():
    """
    Streamlined chat interaction handler with integrated cost tracking
    """
    if not hasattr(st.session_state, 'vector_store'):
        st.warning("Please select a documents folder.")
        return

    display_chat_history()

    if question := st.chat_input("Ask a question about your documents:"):
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.chat_message("user").write(question)

        # Handle assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = []
            
            # Single cost tracker instance for the entire interaction
            cost_tracker = LLMCostTracker()

            try:
                # Calculate prompt cost first
                prompt_messages = st.session_state.chat_history[:-1] + [HumanMessage(content=question)]
                prompt_cost = cost_tracker.calculate_prompt_cost(prompt_messages, config.MODEL_NAME)
                
                # Update session state with prompt cost
                if 'total_llm_cost' not in st.session_state:
                    st.session_state.total_llm_cost = 0.0
                st.session_state.total_llm_cost += prompt_cost

                # Stream response with timeout and cost tracking
                async def process_stream():
                    formatted_history = "\n".join(
                        f"{msg.type}: {msg.content}" 
                        for msg in st.session_state.chat_history[:-1]
                    )
                    
                    async for chunk in st.session_state.chain.astream({
                        "input": question,
                        "chat_history": formatted_history
                    }):
                        content = None
                        if isinstance(chunk, dict):
                            content = (chunk.get("answer") or 
                                     chunk.get("response") or 
                                     chunk.get("output", ""))
                        elif isinstance(chunk, str):
                            content = chunk
                            
                        if content:
                            # Calculate chunk cost
                            chunk_cost = cost_tracker.calculate_streaming_chunk_cost(
                                content, 
                                config.MODEL_NAME
                            )
                            st.session_state.total_llm_cost += chunk_cost
                            
                            # Update display
                            full_response.append(content)
                            response_container.markdown(''.join(full_response))
                            
                            # Update the cost metrics in real-time
                            update_sidebar_metrics()
                            
                            await asyncio.sleep(0.05)  # Smoother rendering

                # Run with timeout
                await asyncio.wait_for(
                    process_stream(),
                    timeout=60
                )

                # Add final response to chat history
                final_response = ''.join(full_response)
                st.session_state.chat_history.append(AIMessage(content=final_response))

                # Get final statistics
                input_tokens, output_tokens = cost_tracker.get_token_counts()
                total_cost = cost_tracker.get_total_cost()

                # Log final costs
                logger.debug(f"""
                    Final cost breakdown:
                    Prompt tokens: {input_tokens}
                    Response tokens: {output_tokens}
                    Prompt cost: ${prompt_cost:.4f}
                    Total cost: ${total_cost:.4f}
                    Response length: {len(final_response)}
                """)

            except asyncio.TimeoutError:
                st.error("Response generation timed out. Please try again.")
                logger.error("Streaming response timed out")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Streaming error: {str(e)}")

def initialize_sidebar():
    """
    Initialize the persistent sidebar structure and containers once at startup
    """
    if 'sidebar_initialized' not in st.session_state:
        with st.sidebar:
            # Create containers for all sidebar elements
            st.session_state.sidebar_header = st.empty()
            st.session_state.sidebar_header.header("📊 Document Insights")
            
            # Document metrics container
            st.session_state.doc_metrics_container = st.empty()
            
            # Cost tracking section with persistent containers
            st.session_state.cost_header = st.empty()
            st.session_state.cost_header.subheader("Cost Tracking")
            
            # Individual metric containers
            st.session_state.embedding_cost = st.empty()
            st.session_state.llm_cost = st.empty()
            st.session_state.total_cost = st.empty()
            
            st.session_state.sidebar_initialized = True

def update_sidebar_metrics():
    """
    Update sidebar metrics using persistent containers
    """
    # Ensure sidebar is initialized
    if 'sidebar_initialized' not in st.session_state:
        initialize_sidebar()
    
    # Update document count using container
    with st.session_state.doc_metrics_container:
        st.metric(
            "Total Documents",
            st.session_state.get('document_count', 0)
        )
    
    # Update costs using persistent containers
    with st.session_state.embedding_cost:
        st.metric(
            "Embedding Cost",
            f"${st.session_state.get('total_embedding_cost', 0.0):.4f}"
        )
    
    with st.session_state.llm_cost:
        st.metric(
            "LLM Message Cost",
            f"${st.session_state.get('total_llm_cost', 0.0):.4f}"
        )
    
    with st.session_state.total_cost:
        total_cost = (
            st.session_state.get('total_embedding_cost', 0.0) + 
            st.session_state.get('total_llm_cost', 0.0)
        )
        st.metric(
            "Total Cost",
            f"${total_cost:.4f}"
        )

def main():
    st.title("Document Chat Assistant")

    # Initialize session state
    for key in ['chat_history', 'documents_loaded', 'previous_folder_path']:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'chat_history' else False

    if not load_environment():
        st.error("Failed to load required environment variables. Please check your .env file.")
        return

    # Initialize sidebar once at startup
    initialize_sidebar()

    documents_folder = select_documents_folder()

    if documents_folder and not st.session_state.documents_loaded:
        with st.spinner("Loading documents... This may take a while."):
            if result := load_text_files_from_folder(documents_folder):
                st.session_state.vector_store, st.session_state.file_monitor = result
                st.session_state.chain = create_streaming_chain(
                    st.session_state.vector_store,
                    config.MODEL_NAME,
                    config.MAX_TOKENS
                )
                st.session_state.documents_loaded = True
                st.success("Documents loaded successfully!")

                # Update metrics after loading documents
                update_sidebar_metrics()
    
    # Handle chat interactions
    asyncio.run(handle_chat_interaction())

if __name__ == "__main__":
    main()