# -RAG-based-Chatbot-Company-Policies-
# Company Policy Chatbot

This is a document-based question-answer chatbot for company policies. It allows users to upload or reference policy documents (PDF/TXT) and ask questions in natural language. The system finds the most relevant content using semantic search and answers each query using state-of-the-art LLM models (Groq API).


## Features

- **Chat UI:** Powered by [Streamlit](https://streamlit.io/) for interactive user experience and conversation history.
- **Semantic search:** Uses [Sentence Transformers](https://www.sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss) for retrieval-augmented context.
- **LLM-based answering:** Utilizes Groq's OpenAI-compatible API for accurate, context-based answers.
- **Memory:** Shows full Q&A history for seamless multi-turn conversations.


## Setup Instructions


### 1. **Install Python Dependencies**

Make sure you’re using Python 3.8+.

install individually:

```bash
pip install streamlit PyPDF2 sentence-transformers faiss-cpu openai python-dotenv pandas
```

### 2. **Add The Policy Documents**

Place company policy PDFs or plain text files in the `data/` directory.

```
data/
  |- IT company policy.pdf
```

### 3. **Configure Groq API Key**

Create a `.env` file in your project’s root directory:

```
GROQ_API_KEY=sk_groq_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 5. **Run the App**

```bash
streamlit run chatbot.py
```
Open the given localhost URL in your browser to use the chatbot.


## Main Approach (How it Works)

1. **Document Loading**  
   All files in the `data/` folder are loaded and parsed.

2. **Chunking**
   Each document is split into manageable text chunks size = 500 characters each.

3. **Embedding & Indexing**
   Chunks are converted to embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
   FAISS builds a vector index for fast similarity search.

4. **Retrieval**
   When you ask a question, the semantic search retrieves the most relevant chunks.

5. **Generation**
   The most relevant context is sent to Groq’s LLM (`llama-3-8b-8192`, `llama-3-70b-8192`, or `gemma-7b-it` as supported). But here i have used 'moonshotai/kimi-k2-instruct'
   The model answers concisely, referencing original sources where possible.

6. **Conversation History**
   All Q&A pairs are shown, enabling natural multi-turn discussions.
