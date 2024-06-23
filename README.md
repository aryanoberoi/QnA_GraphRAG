# Document Assistant

This is a Flask-based web application that serves as a document assistant. It can process various types of documents (PDF, DOC, DOCX, TXT) and URLs, extract text from them, and answer questions based on the content using a combination of Neo4j graph database and the llama3 language model using Ollama.

## Features

- Upload and process multiple document types (PDF, DOC, DOCX, TXT)
- Extract text from URLs
- Store and process text using Neo4j graph database
- Answer questions based on the processed documents using Ollama language model
- Support for multiple languages (input and output)

## Requirements

- Python 3.7+
- Flask
- Neo4j
- Ollama
- Various Python libraries (see requirements.txt)

## Installation

1. Clone the repository:  
```git clone https://github.com/Surya-03/llm-flask.git```

3. Install the required packages:  
```pip install -r requirements.txt```

4. Set up your environment variables in a `.env` file:

5. Ensure Neo4j is running and update the connection details in the code if necessary.

## How to Run

1. Navigate to the project directory:  
```cd llm-flask```

3. Run the Flask application:  
```flask run```

4. Open a web browser and go to `http://127.0.0.1:5000/`

## Function Documentation

### File Processing Functions

- `get_text_from_doc(doc_file)`: Extracts text from a .doc file.
- `get_text_from_docx(docx_file)`: Extracts text from a .docx file.
- `get_text_from_txt(txt_file)`: Extracts text from a .txt file.
- `get_text_from_pdf(pdf_file)`: Extracts text from a PDF file.
- `get_text_from_files(files)`: Processes multiple files and extracts text from them.

### URL Processing Function

- `get_text_from_url(url)`: Extracts text content from a given URL.

### Text Processing Functions

- `get_text_chunks(text)`: Splits the input text into smaller chunks for processing.
- `get_vector_store(text_chunks, usersession)`: Creates vector representations of text chunks and stores them in the Neo4j graph database.

### Conversation Chain Functions

- `get_conversational_chain()`: Sets up the retrieval QA chain using Neo4j and Ollama.
- `user_input(user_question)`: Processes user input and generates a response using the conversational chain.

### Flask Routes

- `index()`: Handles the main page, file uploads, and URL processing.
- `ask()`: Handles user questions and generates responses.

## Main Application Flow

1. The user uploads documents or provides a URL.
2. The application extracts text from the uploaded files or URL.
3. The extracted text is split into chunks and stored in the Neo4j graph database.
4. The user can then ask questions about the processed documents.
5. The application uses the Ollama language model to generate responses based on the stored information.
6. If needed, the response is translated to the user's preferred language.

## Note

Make sure to have Neo4j running and properly configured before starting the application. Also, ensure that the Ollama model is properly set up and accessible.

For any issues or questions, please open an issue in the GitHub repository.
