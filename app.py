import os
# Suppress InsecureRequestWarning in this process and any subprocesses (e.g. unstructured)
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"

import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import validators
import streamlit as st  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Streamlit app
st.set_page_config(page_title="Text Summarization from Youtube or website", page_icon=":book:", layout="wide")
st.title("Text Summarization from Youtube or website")
st.subheader("Enter the URL of the Youtube video or website to summarize the text")

# Session state: summary history per session_id (only current session shown in UI)
if "summary_history" not in st.session_state:
    st.session_state.summary_history = {}  # {session_id: [{"url": ..., "summary": ...}, ...]}
elif isinstance(st.session_state.summary_history, list):
    # Migrate old flat list to per-session dict (assign all to default_session)
    st.session_state.summary_history = {"default_session": st.session_state.summary_history}

# Session state for LLM chat history (per session_id)
if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return chat message history for the given session_id (used by the LLM)."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


## Get the Groq API key
groq_api_key =""
with st.sidebar:
    groq_api_key_input = st.text_input("Enter your Groq API Key:", type="password")
    if groq_api_key_input:
        groq_api_key = groq_api_key_input
    else:
        st.error("Missing  Groq API Key ")

#groq_api_key = os.getenv("GROQ_API_KEY_1")

## Session ID for LLM conversation history
with st.sidebar:
    session_id = st.text_input("Session ID (for LLM history)", value="default_session", help="Same ID = same conversation context for the model")

## Get the URL of the Youtube video or website
genric_url = st.text_input("URL: Enter the URL of the Youtube video or website:", label_visibility="collapsed")


### llm
if groq_api_key_input:
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)


    # Chunk size to stay under Groq token limit (~6000 TPM). ~1 token ≈ 4 chars, so ~4000 chars is safe per call.
    CHUNK_CHARS = 4000
    CHUNK_OVERLAP = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)

    # Prompt for summarizing a single chunk (no history, kept small)
    chunk_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this text in a few sentences. Capture the main points only."),
        ("human", "{content}"),
    ])
    chunk_chain = chunk_prompt | llm | StrOutputParser()

    # Final combine step: merge chunk summaries into one; includes session history
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You combine the following partial summaries into one coherent summary of about 300 words. Be concise and capture the main points. If there is prior conversation, you can refer to it for context."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Partial summaries to combine:\n\n{content}"),
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()

else:
    st.error("Missing  Groq API Key ")


if st.button("Summarize the content"):
    ### validate input
    #if not groq_api_key_input.strip() or not genric_url.strip():
        #st.error("Please enter a valid Groq API key and URL")
        #st.stop()
    if not validators.url(genric_url):
        st.error("Please enter a valid URL")
        st.stop()
    else :
        try:
            with st.spinner("Loading the data..."):
                ## Loading the url data
                if "youtube.com" in genric_url:
                    loader = YoutubeLoader.from_youtube_url(genric_url, add_video_info=False)
                    docs = loader.load()
                else:
                    # Use browser-like headers to reduce 400/403 from strict sites
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    }
                    loader = UnstructuredURLLoader(urls=[genric_url], ssl_verify=True, headers=headers)
                    docs = loader.load()
                st.write(f"Documents count: {len(docs)}")
                content = "\n\n".join(d.page_content for d in docs)
                session_history = get_session_history(session_id)
                # Split into chunks to stay under Groq token limit (413 = request too large)
                chunks = text_splitter.split_text(content)
                if not chunks:
                    st.error("No text could be extracted from the URL.")
                    st.stop()
                # Step 1: summarize each chunk (small requests)
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    chunk_summaries.append(chunk_chain.invoke({"content": chunk[:CHUNK_CHARS]}))
                combined = "\n\n---\n\n".join(chunk_summaries)
                # Step 2: combine into one summary (with session history for context)
                output_summary_chain = summarize_chain.invoke({
                    "chat_history": session_history.messages,
                    "content": combined,
                })
                # Add this turn to LLM session history (short user msg to avoid huge history)
                session_history.add_user_message(f"Summarize content from: {genric_url}")
                session_history.add_ai_message(output_summary_chain)
                # Add to UI summary history for this session only
                if session_id not in st.session_state.summary_history:
                    st.session_state.summary_history[session_id] = []
                st.session_state.summary_history[session_id].append({
                    "url": genric_url,
                    "summary": output_summary_chain,
                })
                st.success("Summary generated.")
                st.write(output_summary_chain)
        except Exception as e:
            err_msg = str(e)
            if "400" in err_msg or "Bad Request" in err_msg:
                st.error(
                    "Error loading the data: The server returned 400 Bad Request. "
                    "Some sites block automated access. Try a different URL."
                )
            else:
                st.error(f"Error loading the data: {e}")
            with st.expander("Error details"):
                st.code(err_msg)
            st.stop()

# Show session history in sidebar
with st.sidebar:
    st.subheader("Session history")
    _hist = get_session_history(session_id)
    if _hist.messages:
        st.caption(f"LLM sees {len(_hist.messages)} messages in this session.")
    if st.button("Clear LLM history for this session"):
        if session_id in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    st.divider()
    # Only show summaries for the current session_id
    current_summaries = st.session_state.summary_history.get(session_id, [])
    if not current_summaries:
        st.caption("No summaries in this session. Summarize a URL to see them here.")
    else:
        for entry in reversed(current_summaries):
            with st.expander(f"📎 {entry['url'][:50]}..." if len(entry["url"]) > 50 else f"📎 {entry['url']}"):
                st.write(entry["summary"])
        if st.button("Clear summary list"):
            st.session_state.summary_history[session_id] = []
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
