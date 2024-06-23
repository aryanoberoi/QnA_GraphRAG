# Required imports
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import asyncio
from PyPDF2 import PdfReader
from docx import Document
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from neo4j import GraphDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain.graphs import Neo4jGraph
from langchain.docstore.document import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Neo4j connection
URI = "bolt://localhost:7687"
graph = Neo4jGraph(url=URI, username="neo4j", password="password")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to handle asyncio event loop
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

get_or_create_eventloop()

# Functions to extract text from different file types
def get_text_from_doc(doc_file):
    document = Document(doc_file)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def get_text_from_docx(docx_file):
    temp_file_path = "temp.docx"
    with open(temp_file_path, "wb") as f:
        f.write(docx_file.read())
    return docx2txt.process(temp_file_path)

def get_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def get_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from multiple files
def get_text_from_files(files):
    text = ""
    for file in files:
        if file.filename.endswith(".pdf"):
            text += get_text_from_pdf(file)
        elif file.filename.endswith(".doc"):
            text += get_text_from_doc(file)
        elif file.filename.endswith(".docx"):
            text += get_text_from_docx(file)
        elif file.filename.endswith(".txt"):
            text += get_text_from_txt(file)
        else:
            return f"Unsupported file type: {file.filename}"
    return text

# Function to extract text from a URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except Exception as e:
        return f"Error fetching the URL: {e}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    return text_splitter.split_text(str(text[:3]))

# Function to create and store vector representations of text chunks
def get_vector_store(text_chunks, usersession):
    text_chunks = get_text_chunks(text_chunks)
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    llm = OllamaFunctions(model="llama3")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

# Function to set up the conversational chain
def get_conversational_chain():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        url="bolt://localhost:7687",
        username='neo4j',
        password='password',
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    try:
        llm = ChatOllama(model="llama3", format="json", temperature=0)
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return None

    vector_qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_index.as_retriever()
    )
    return vector_qa

# Function to process user input and generate a response
def user_input(user_question):
    vector_qa = get_conversational_chain()
    if vector_qa is None:
        return {"output_text": "Error: Could not load the model."}
    response = vector_qa.run(user_question)
    return response

# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    file_details = []
    url_displayed = session.get('url_input', '')

    if 'session_id' not in session:
        session['session_id'] = os.urandom(24).hex()
        session['chat_history'] = [
            AIMessage(content="Hello! I'm a document assistant. Ask me anything about the documents you upload."),
        ]

    if request.method == 'POST':
        files = request.files.getlist("files")
        url_input = request.form.get("url_input")
        raw_text = ""
        session["input_language"] = int(request.form.get("input_language"))
        session["output_language"] = int(request.form.get("output_language"))
        
        # Process uploaded files
        if files and files[0].filename != '':
            valid_files = all(f.filename.endswith(('.pdf', '.doc', '.docx', '.txt')) for f in files)
            if valid_files:
                raw_text += get_text_from_files(files)
                message = "Files successfully uploaded."
                for file in files:
                    file_details.append({"name": file.filename})
            else:
                message = "Please upload files in PDF, DOC, DOCX, or TXT format."

        # Process URL input
        if url_input:
            url_text = get_text_from_url(url_input)
            raw_text += " " + url_text
            message = f"Files and URL processed successfully. URL: {url_input}"
            session['url_input'] = url_input

        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, session['session_id'])

    chat_history = session.get('chat_history', [])
    return render_template('index.html', chat_history=chat_history, message=message, file_details=file_details, url_displayed=url_displayed)

# Route for handling user questions
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("question")
    if user_query and user_query.strip():
        session['chat_history'].append(HumanMessage(content=user_query))
        response = user_input(user_query)

        res = response
        session['chat_history'].append(AIMessage(content=res))

        # Translate the response if needed
        if int(session["output_language"]) != 23:
            payload = {
                "source_language": session["input_language"],
                "content": res,
                "target_language": session["output_language"]
            }
            res = json.loads(requests.post('http://127.0.0.1:8000/scaler/translate', json=payload).content)
            res = res['translated_content']

        return jsonify({"answer": res, "url": session.get('url_input', '')})
    return jsonify({"error": "Invalid input"})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
