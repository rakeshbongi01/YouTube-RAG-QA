import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# 1. Load environment variables from the .env file
load_dotenv()

# Optional: Add a safety check to fail early if the key is missing
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

def format_docs(retrieved_docs):
    """Helper function to format retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs) 

def run_rag_pipeline(video_id: str, query: str):
    print(f"Fetching transcript for video ID: {video_id}...")
    try:
        # Step 1: Ingestion
        # Fetch the data and convert it back to raw dictionaries immediately 
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"]).to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        # Step 2: Text Splitting
        print("Chunking transcript...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript]) # Same as document_loaders

        # Step 3: Embeddings & Vector Store
        print("Generating embeddings and building FAISS index...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Step 4: Retriever Setup
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        """
        as_retriever(): Converts the static vector database into an active LangChain interface that automatically embeds string queries and executes the search.
        search_type="similarity": Retrieves documents using nearest-neighbor algorithms based on semantic mathematical closeness rather than exact keyword matches.
        search_kwargs={"k": 4}: Restricts the output to the top 4 most relevant chunks to optimize API costs and protect the LLM's context window.
        """

        # Step 5: LLM & Prompt Configuration
        llm = ChatOpenAI(model="gpt-5o-mini", temperature=0.2)
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

        # Step 6: Build the LCEL (LangChain Expression Language) Chain
        parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs), #[cite: 1]
                    'question': RunnablePassthrough() #[cite: 1]
                })
        """
        RunnableParallel builds the exact dictionary required by the PromptTemplate by splitting the single user query into two concurrent paths[cite: 1]:
        - 'context': Pipes the query into the retriever to fetch relevant chunks, then formats them into a single string[cite: 1].
        - 'question': Uses RunnablePassthrough to carry the exact user query forward unaltered[cite: 1].
        """ 

        main_chain = parallel_chain | llm | prompt | StrOutputParser()


        # Step 7: Execution
        print(f"\nExecuting Query: '{query}'")
        answer = main_chain.invoke(query)

    except TranscriptsDisabled:
        print("Error: No captions available for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Test parameters from your notebook
    target_video = "k3O-TL4riQQ"
    test_question = "Can you summarize the video in english?"
    
    run_rag_pipeline(video_id=target_video, query=test_question)