import streamlit as st
import PyPDF2
import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from openai import OpenAI

def read_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def load_documents(folder_path):
    docs = []
    for file in glob.glob(os.path.join(folder_path, "*")):
        if file.lower().endswith(".pdf"):
            docs.append({'filename': os.path.basename(file), 'text': read_pdf(file)})
        elif file.lower().endswith(".txt"):
            with open(file, 'r', encoding='utf-8') as f:
                docs.append({'filename': os.path.basename(file), 'text': f.read()})
    return docs

documents = load_documents("data") 


def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

all_chunks = []
for doc in documents:
    for chunk in chunk_text(doc['text']):
        all_chunks.append({'filename': doc['filename'], 'chunk': chunk})

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunk_texts = [c['chunk'] for c in all_chunks]
embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def search_chunks(query, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    results = []
    for idx in I[0]:
        results.append(all_chunks[idx])
    return results

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

def generate_answer(context_chunks, user_query):
    context = " ".join([f"[{c['filename']}] {c['chunk']}" for c in context_chunks])
    prompt = (
        "Based only on this company policy context below, answer the user's query in a concise and neutral way. "
        "Cite the source document in your answer.\n\n"
        f"Context: {context}\n\nQuery: {user_query}\n\nAnswer:"
    )
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content

st.title("Company Policy Chatbot")

if "history" not in st.session_state:
    st.session_state["history"] = []
if "user_q" not in st.session_state:
    st.session_state["user_q"] = ""

def process_query():
    user_query = st.session_state["user_q"]
    if user_query:
        hits = search_chunks(user_query)
        answer = generate_answer(hits, user_query)
        st.session_state["history"].append({"question": user_query, "answer": answer})
        st.session_state["user_q"] = "" 

for turn in st.session_state["history"]:
    st.markdown(f"**You:** {turn['question']}")
    st.markdown(f"**Bot:** {turn['answer']}")

st.text_input(
    "Ask a company policy question:",
    key="user_q",
    on_change=process_query
)
    