import streamlit as st
import logging
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Updates", page_icon="🦙", layout="centered", initial_sidebar_state="auto")

# Set OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Initialize chat memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


# Streamlit title
st.title("Hi There!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How May I Help?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(directory="./docs", model="gpt-3.5-turbo", temperature=0):
    """Load and index documents for use with OpenAI model."""
    try:
        with st.spinner(text="Loading and indexing the – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(model=model, temperature=temperature, system_prompt="You are an virtual assistant for instruqt documentation and can help in creating tracks and challenges. Answer from provided documents only.")
            )
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            logger.info("Data loading and indexing completed successfully.")
            return index
    except Exception as e:
        logger.error(f"Error during data loading and indexing: {e}")
        st.error("Failed to load and index data. Please check the logs for more details.")

# Load data and initialize chat engine
index = load_data()
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", memory=memory, system_prompt="You are an virtual assistant for instruqt documentation and can help in creating tracks and challenges. Answer from provided documents only. ", verbose=True
    )

# Chat input and response handling
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response if the last message is from the user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
