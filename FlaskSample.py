from flask import Flask, request, render_template
from flask_mysqldb import MySQL
"""
from pathlib import Path
from typing import List, Tuple
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field
from langchain.chains import ConversationalRetrievalChain
# Constants
local_path = "./models/gpt4all-converted.bin"
model_path = "./models/ggml-model-q4_0.bin"
text_path = "./docs/file0.txt"
index_path = "./indexes"

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = TextLoader(text_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path, max_tokens =2048, verbose=True)

embeddings = initialize_embeddings()
sources = load_documents()
chunks = split_chunks(sources)
vectorstore = generate_index(chunks, embeddings)
vectorstore.save_local("full_sotu_index")

index = FAISS.load_local(index_path, embeddings)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)
chat_history=[]

"""
app  = Flask(__name__)
@app.route('/')
def home():
    return  render_template("index.html")


@app.route("/get", methods=["GET","POST"])
def chatbot_response():
    return "Not connected to chatbot"
    
"""
    msg = request.form["msg"]
    response = qa({"question": msg, "chat_history": chat_history})
    return response['answer']"""

@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/about')
def about():
    return render_template("about.html")

if __name__=="__main__":
    app.run(debug=True)