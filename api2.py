from fastapi import FastAPI, UploadFile, File
import PyPDF2
import pandas as pd
import textwrap
from pydantic import BaseModel
from cdb import upsert_documents, query_collection
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

from pydantic import BaseModel

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)


class QueryModel(BaseModel):
    query: str

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    all_chunks = []

    for file in files:
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "pdf":
            # Process PDF file
            pdf_text = ''
            reader = PyPDF2.PdfReader(file.file)
            for page in reader.pages:
                pdf_text += page.extract_text() or ''
            chunks = textwrap.wrap(pdf_text, 1000)

        elif file_extension == "csv":
            # Process CSV file
            df = pd.read_csv(file.file)
            csv_text = df.to_string(index=False)  # Convert CSV to text
            chunks = textwrap.wrap(csv_text, 1000)

        else:
            return {"error": f"Unsupported file type: {file.filename}"}

        all_chunks.extend(chunks)

    # Store all extracted chunks into ChromaDB
    upsert_documents(documents=all_chunks)
    
    return {"message": "Files uploaded and processed", "total_chunks": len(all_chunks)}

@app.post("/query/")
async def query_data(query_data: QueryModel):
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
