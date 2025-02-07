import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.title('PDF Question Answering System')

# Upload PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    files_to_send = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
    
    with st.spinner("Uploading and processing..."):
        response = requests.post(f"{API_URL}/upload/", files=files_to_send)
    
    if response.status_code == 200:
        st.success("PDFs processed successfully!")
    else:
        st.error("Error uploading PDFs.")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the PDF content")

if query:
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/query/", json={"query": query})
    
    if response.status_code == 200:
        answer = response.json()["response"]
        st.chat_message("assistant").write(answer)
    else:
        st.error("Error fetching response.")
