from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import asyncio
from PyPDF2 import PdfReader
from docx import Document
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
import os
import json
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import pandas as pd
import spacy
import requests
from bs4 import BeautifulSoup
import PyPDF2
from neo4j import GraphDatabase
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph  # Added import
from langchain_community.llms import Ollama  # Added import
from langchain.text_splitter import TokenTextSplitter  # Added import
from langchain_experimental.graph_transformers import LLMGraphTransformer  # Added import
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector  # Added import

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain.docstore.document import Document 
'''part of the LangChain library and is used to represent a piece of text or document data along with its metadata. 
It is a common data structure used throughout the LangChain library for storing and processing text data.
In your code, you are using the Document class to create new Document objects from the preprocessed text data. 
Specifically, you are creating a dictionary with the page_content (the actual text content) and metadata (additional information about the text, such as the source), 
and then creating a Document object using that dictionary.'''
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from py2neo import Graph
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama  # Added import
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser
from pydantic import BaseModel
from typing import List
from langchain.chains import RetrievalQA  # Added import
from langchain_community.embeddings import OllamaEmbeddings
from tempfile import NamedTemporaryFile
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
from openai import OpenAI
URI = "neo4j+s://0479a26a.databases.neo4j.io"
from pyvis.network import Network
graph = Neo4jGraph(url=URI, username="neo4j", password="OZcxm3oDjT8I0WRxC2RgkqAudZOnZ174ciaDk2gQuSM")  # Initialized Neo4j Graph

# Ensure the environment variables are loaded
load_dotenv()
docs=[]
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

def get_text_from_doc(doc_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(doc_file.filename))
    
    try:
        doc_file.save(temp_file_path)
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(temp_file_path)
        data = loader.load()
        print(data)
        return data
    finally:
        os.remove(temp_file_path)

def get_text_from_txt(doc_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(doc_file.filename))
    
    try:
        doc_file.save(temp_file_path)
        from langchain_community.document_loaders.text import TextLoader

        loader = TextLoader(temp_file_path)
        data = loader.load()
        print(data)
        return data
    finally:
        os.remove(temp_file_path)
def get_text_from_pdf(pdf_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
    pdf_file.save(temp_file_path)
    from langchain_community.document_loaders import PyPDFLoader


    loader = PyPDFLoader(temp_file_path)

    data = loader.load()
    print(data)
    return data

def get_text_from_files(files):
    for file in files:
        if file.filename.endswith(".pdf"):
            docs.extend(get_text_from_pdf(file))
        elif file.filename.endswith(".doc"):
            docs.extend(get_text_from_doc(file))
        elif file.filename.endswith(".docx"):
            docs.extend(get_text_from_doc(file))
        elif file.filename.endswith(".txt"):
            docs.extend(get_text_from_txt(file))
        else:
            return f"Unsupported file type: {file.filename}"
    return docs



def get_text_from_url(urls):
   loader = UnstructuredURLLoader(urls=urls)
   return loader.load()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n\n")
    basic = text_splitter.split_documents(text)  # Splitting text into chunks

  
    return basic

def get_graph(graph_documents):
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        
        nodes = graph_documents[0].nodes
        relationships = graph_documents[0].relationships
        
        for node in nodes:
            net.add_node(node.id, label=node.id, title=str(node.type), color="skyblue")
        
        for relationship in relationships:
            net.add_edge(relationship.source.id, relationship.target.id, title=relationship.type, color="gray", arrows="to")
        
        net.repulsion()
        
        # Generate HTML file
        net.write_html("graph.html")

#get_graph()

# Added get_vector_store function
def get_vector_store(text_chunks, usersession):
    # Preprocess the documents to convert lists to tuples
    text_chunks = get_text_chunks(text_chunks)

    # Create Document objects from the text chunks

    #llm = OllamaFunctions(model="phi3")
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(text_chunks)
    
    graph.add_graph_documents (
      graph_documents
    )
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")


    graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
        ...,
            description="All the person, organization, or business entities that "
        "appear in the text",
    )


# Added get_conversational_chain function
def get_conversational_chain():
    embeddings = OpenAIEmbeddings()
    #OllamaEmbeddings(model="mxbai-embed-large")
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        url="neo4j+s://0479a26a.databases.neo4j.io",
        username='neo4j',
        password='OZcxm3oDjT8I0WRxC2RgkqAudZOnZ174ciaDk2gQuSM',
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    try:
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return None

    vector_qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_index.as_retriever()
    )
    return vector_qa

# Added user_input function
def user_input(user_question):
    vector_qa = get_conversational_chain()
    if vector_qa is None:
        return {"output_text": "Error: Could not load the model."}
    response = vector_qa.run(user_question)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None  # Initialize a message variable
    file_details = []  # Initialize a list to store file details
    url_displayed = session.get('url_input', '')  # Retrieve the stored URL or set it to empty string

    if 'session_id' not in session:
        session['session_id'] = os.urandom(24).hex()
        session['chat_history'] = [
            AIMessage(content="Hello! I'm a document assistant. Ask me anything about the documents you upload."),
        ]

    if request.method == 'POST':
        files = request.files.getlist("files")
        url_input = request.form.get("newInput")
        raw_text = []
        session["input_language"] = int(request.form.get("input_language"))
        
        session["output_language"] = int(request.form.get("output_language"))
        # Process files
        if files and files[0].filename != '':
            valid_files = all(f.filename.endswith(('.pdf', '.doc', '.docx', '.txt')) for f in files)
            if valid_files:
                raw_text.extend( get_text_from_files(files))
                message = "Files successfully uploaded."

                # Get file details for display
                for file in files:
                    file_details.append({"name": file.filename})
            else:
                message = "Please upload files in PDF, DOC, DOCX, or TXT format."

        # Process URL
        if url_input:
            url_text = get_text_from_url(url_input)
            raw_text.extend(url_text)  # Concatenate URL text with existing text
             # Debug print to check what is being added
            message = "Files and URL processed successfully. URL : "+ url_input

            session['url_input'] = url_input  # Store the URL in the session


        if raw_text:
            get_vector_store(raw_text, session['session_id'])  # Call get_vector_store function

    chat_history = session.get('chat_history', [])
    return render_template('index.html', chat_history=chat_history, message=message, file_details=file_details, url_displayed=url_displayed)


@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("question")
    if user_query and user_query.strip():
        session['chat_history'].append(HumanMessage(content=user_query))
        response = user_input(user_query)  # Call user_input function

        res = response
        session['chat_history'].append(AIMessage(content=res))
        print(request.form.get("input_language"))
        

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
@app.route('/getgraph')
def show_graph():
    return render_template('graph.html')


if __name__ == "__main__":
    app.run(debug=True, port=8080)
