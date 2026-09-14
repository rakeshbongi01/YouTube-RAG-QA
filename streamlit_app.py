import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")

# Production-grade minimal styling (No emojis, clean modern palette)
st.markdown("""
<style>
    /* Base typography & header adjustments */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
    
    .app-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .app-subheader {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Primary button: clean slate/indigo fill */
    div.stButton > button:first-child {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid #0f172a;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.15s ease-in-out;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        color: #ffffff;
    }

    /* Structured response card */
    .response-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #2563eb;
        border-radius: 6px;
        padding: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

st.markdown('<div class="app-header">YouTube Transcript Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subheader">Ground LLM responses strictly in indexed video transcripts via FAISS & LangChain.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Authentication")
    api_key = st.text_input("OpenAI API Key", type="password", help="Key is stored in session memory only.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.caption("Ephemeral session. Credentials are discarded on tab close.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Source Video")
    video_id = st.text_input("Video ID", placeholder="e.g. k3O-TL4riQQ")
    
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

with col2:
    st.markdown("### Query Pipeline")
    query = st.text_input("Question", placeholder="Ask anything about the video content...")
    
    if st.button("Generate Answer"):
        if not api_key:
            st.error("Missing OpenAI API Key in configuration panel.")
        elif not video_id:
            st.warning("Please supply a valid YouTube Video ID.")
        elif not query:
            st.warning("Query prompt cannot be empty.")
        else:
            if st.session_state.current_video_id != video_id:
                try:
                    with st.spinner("Ingesting transcript and compiling FAISS index..."):
                        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"]).to_raw_data()
                        transcript = " ".join(chunk["text"] for chunk in transcript_list)

                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = splitter.create_documents([transcript])

                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                        vector_store = FAISS.from_documents(chunks, embeddings)

                        st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                        st.session_state.current_video_id = video_id
                        
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled or unavailable for this video.")
                    st.stop()
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
                    st.stop()

            with st.spinner("Retrieving relevant chunks and generating completion..."):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                prompt = PromptTemplate(
                    template="""
                      You are a helpful assistant.
                      Answer ONLY from the provided transcript context.
                      If the context is insufficient, just say you don't know.

                      {context}
                      Question: {question}
                    """,
                    input_variables=['context', 'question']
                )

                parallel_chain = RunnableParallel({
                    'context': st.session_state.retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                main_chain = parallel_chain | prompt | llm | StrOutputParser()

                answer = main_chain.invoke(query)
                
                st.markdown("### Response")
                st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)