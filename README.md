# Document Assistant with Bhashini Integration

This is a Flask-based web application that serves as a document assistant with integrated Bhashini translation services. It processes various document types, extracts text, answers questions using Neo4j and Ollama, and translates responses to multiple Indic languages.

## Features

- Process multiple document types (PDF, DOC, DOCX, TXT)
- Extract text from URLs
- Store and process text using Neo4j graph database
- Answer questions using Ollama language model
- Translate responses to various Indic languages via Bhashini

## Requirements

- Python 3.7+
- Flask
- FastAPI
- Neo4j
- Ollama
- Other dependencies (see requirements.txt)

## Installation

1. Clone the repository:  
```git clone https://github.com/Surya-03/QnA_GraphRAG.git```  
```cd document-assistant```
3. Install required packages:  

```pip install -r requirements.txt```

5. Set up `.env` file

6. Ensure Neo4j is running and configured correctly.

## How to Run

1. Start the Flask application:  
```flask run```  
Access at `http://127.0.0.1:5000/`

3. Run Bhashini translation service:  
```python -m uvicorn custom_api:app --reload```

## Function Documentation

### File Processing Functions

- `get_text_from_doc(doc_file)`: Extract text from .doc files
- `get_text_from_docx(docx_file)`: Extract text from .docx files
- `get_text_from_txt(txt_file)`: Extract text from .txt files
- `get_text_from_pdf(pdf_file)`: Extract text from PDF files
- `get_text_from_files(files)`: Process multiple files and extract text

### URL Processing Function

- `get_text_from_url(url)`: Extract text content from a given URL

### Text Processing Functions

- `get_text_chunks(text)`: Split input text into smaller chunks
- `get_vector_store(text_chunks, usersession)`: Create vector representations and store in Neo4j

### Conversation Chain Functions

- `get_conversational_chain()`: Set up retrieval QA chain using Neo4j and Ollama
- `user_input(user_question)`: Process user input and generate responses

### Flask Routes

- `index()`: Handle main page, file uploads, and URL processing
- `ask()`: Handle user questions and generate responses

## Bhashini Integration

The Bhashini integration allows translation of LLM responses into various Indic languages.

### Features:

- Support for multiple Indian languages and English
- High-quality translations via Bhashini API
- Implemented as a separate FastAPI application (`custom_api.py`)

### Functionality:

1. Exposes `/scaler/translate` endpoint
2. Accepts translation requests (source language, content, target language)
3. Communicates with Bhashini API to:
a. Retrieve translation model pipeline
b. Perform translation
4. Returns translated content to main application

### Supported Languages:

23 languages including Hindi, Tamil, Telugu, Bengali, etc., each assigned a numeric code.

## Main Application Flow

1. User uploads documents or provides URL
2. Application extracts text from files/URL
3. Text is chunked and stored in Neo4j graph database
4. User asks questions about processed documents
5. Ollama language model generates responses
6. Responses translated to preferred language using Bhashini integration (if needed)

## Note

Ensure Neo4j and Ollama are properly set up and running. For Bhashini integration, verify you have the necessary API keys and permissions.

For issues or questions, please open an issue in the GitHub repository.

To install dependencies:  
```pip install -r requirements.txt```
