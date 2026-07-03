import os

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

video_id = "etnLX7m2MiA"

api = YouTubeTranscriptApi()

fetched = api.fetch(
    video_id,
    languages=["hi"]
)

transcript = " ".join(item.text for item in fetched)
    
    
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
chunks = splitter.create_documents([transcript])

print("Total chunks:",len(chunks))


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks,embeddings)



retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})


llm = ChatGroq(model ="llama-3.3-70b-versatile",temperature =0.1)

prompts = PromptTemplate(
    template = """You are a helpful assistant.

Answer ONLY from the provided transcript.

If the answer is not present in the transcript, simply say:

"I don't know."

Context:

{context}

Question:

{question}

""",
input_variables = ['context', 'question']
)

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)
    
parallel_chain = RunnableParallel({
    "context":retriever| RunnableLambda(format_docs),
    "question":RunnablePassthrough()
    })


parser = StrOutputParser()


main_chain = parallel_chain | prompts | llm | parser

r = main_chain.invoke("kiya ise video main ai ki baat hui hai?")

print(r)

