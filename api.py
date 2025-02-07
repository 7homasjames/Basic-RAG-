from fastapi import FastAPI, UploadFile, File
import PyPDF2
import textwrap
from pydantic import BaseModel
from cdb import upsert_documents, query_collection
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Initialize Chat Model
chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model='gpt-3.5-turbo')
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Store uploaded documents
stored_chunks = []

class QueryModel(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global stored_chunks
    all_chunks = []
    
    for file in files:
        pdf_text = ''
        reader = PyPDF2.PdfReader(file.file)
        for page in reader.pages:
            pdf_text += page.extract_text() or ''
        
        # Process text into chunks
        chunk_size = 1000
        chunks = textwrap.wrap(pdf_text, chunk_size)
        all_chunks.extend(chunks)
    
    stored_chunks = all_chunks  # Save chunks for querying
    upsert_documents(documents=all_chunks)
    
    return {"message": "Files uploaded and processed", "total_chunks": len(all_chunks)}

@app.post("/query/")
async def query_pdf(query_data: QueryModel):
    query_results = query_collection(query_texts=[query_data.query], n_results=3)

    try:
        source_knowledge = "\n".join([doc for result in query_results['documents'] for doc in result])
    except KeyError:
        source_knowledge = "No relevant context found."

    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query_data.query}"""

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=augmented_prompt)
    ]

    response = chat(messages)
    return {"response": response.content}
