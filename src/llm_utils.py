from typing import Iterator, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def create_embeddings():
    """Create and return OpenAI embeddings"""
    return OpenAIEmbeddings()

def create_chat_model(model_name: str, max_tokens: int = 4000):
    """
    Create and return a chat model based on the specified model name
    
    Args:
        model_name (str): Name of the model to use (e.g., 'claude-3-opus-20240229')
        max_tokens (int): Maximum tokens in the response
    
    Returns:
        ChatAnthropic: Configured chat model
    """
    return ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        streaming=True
    )

def create_streaming_chain(vector_store, model_name: str, max_tokens: int = 4000):
    """
    Create a streaming-enabled retrieval chain
    
    Args:
        vector_store: FAISS vector store
        model_name (str): Name of the model to use
        max_tokens (int): Maximum tokens in the response
    
    Returns:
        Chain: Configured retrieval chain
    """
    model = create_chat_model(model_name, max_tokens)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
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
def query_llm_stream(
    chain: ConversationalRetrievalChain, 
    query: str, 
    chat_history: list[BaseMessage]
) -> Iterator[Dict[str, str]]:
    """
    Stream responses from the LLM with retry capability
    
    Args:
        chain: The conversational chain to use
        query: The user's question
        chat_history: List of previous messages
    
    Yields:
        Dict containing chunks of the response as they become available
    """
    try:
        response = chain.stream({
            "input": query,
            "chat_history": chat_history
        })
        
        for chunk in response:
            if 'answer' in chunk:
                yield {"answer": chunk['answer']}
                
    except Exception as e:
        yield {"error": f"Error generating response: {str(e)}"}