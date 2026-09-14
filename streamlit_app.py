import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Configure the Streamlit page layout
st.set_page_config(page_title="YouTube RAG QA", layout="wide")

def format_docs(retrieved_docs):
    """Helper function to format retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Initialize Session State variables to store data between button clicks
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

st.title("YouTube Transcript Q&A")

# Sidebar for secure API key entry
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Use a side-by-side layout: Video on the left, Q&A on the right
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Load Video")
    video_id = st.text_input("Enter YouTube Video ID (e.g., k3O-TL4riQQ):")
    
    if st.button("Process Video"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not video_id:
            st.warning("Please enter a Video ID.")
        else:
            try:
                with st.spinner("Fetching transcript and building vector index..."):
                    # Step 1: Ingestion
                    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"]).to_raw_data()
                    transcript = " ".join(chunk["text"] for chunk in transcript_list)

                    # Step 2: Text Splitting
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.create_documents([transcript])

                    # Step 3: Embeddings & Vector Store
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vector_store = FAISS.from_documents(chunks, embeddings)

                    # Step 4: Retriever Setup
                    # Save the retriever to session state so it persists for questions
                    st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                    st.session_state.current_video_id = video_id
                    
                    st.success("Video processed successfully!")
            except TranscriptsDisabled:
                st.error("No captions available for this video.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
    # Embed the YouTube video automatically if processing was successful
    if st.session_state.current_video_id:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.current_video_id}")

with col2:
    st.subheader("2. Ask Questions")
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if not st.session_state.retriever:
            st.warning("Please process a video first.")
        elif not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing transcript..."):
                # Step 5: LLM & Prompt Configuration
                # Note: Corrected model name to standard gpt-4o-mini
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

                # Step 6: Build the LCEL Chain
                parallel_chain = RunnableParallel({
                    'context': st.session_state.retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                # Note: The LCEL sequence MUST pipe into the prompt before the LLM
                main_chain = parallel_chain | prompt | llm | StrOutputParser()

                # Step 7: Execution
                answer = main_chain.invoke(query)
                
                st.markdown("**Answer:**")
                st.info(answer)