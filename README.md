# Conversational Study Assistant

This project is built to build a bot that can analyze various files and answer any queries related to them. By the time of finishing the bot will be able to read and analyze files like `.txt`, `.docx`, `.csv`, `.pdf` and, etc. I have used Google's Gemini and used langchain framework for building this application. 

- Langchain is an open-source framework used to build LLM applications. It consists of various tools and abstractions to help developers build various AI applications.
- To learn more, you can read the documentation of [Langchain](https://python.langchain.com/v0.2/docs/introduction/)
## Features

### 1. PDF Q&A Bot
- **Upload PDFs:** Users can upload multiple PDF files which are then processed to extract text.
- **Text Chunks & Vector Store:** The extracted text is split into manageable chunks and stored in a vector database using FAISS and Google Generative AI Embeddings.
- **Conversational Q&A:** Users can ask questions based on the content of the uploaded PDFs. The model searches through the processed text to find relevant information and answers the questions.
- **Conversation History:** Keeps track of the conversation history, which can be cleared or downloaded as a text file.

### 2. Chatbot
- **Generative AI Model:** Utilizes Google's Gemini-Pro model for a conversational experience.
- **Interactive Chat:** Users can chat with the AI, asking it various questions or engaging in a general conversation.
- **Session History:** Chat history is preserved within the session and can be reviewed during the conversation.

## Installation

### Prerequisites
- Python 3.8+
- `Streamlit`
- `langchain_google_genai` package
- `google.generativeai`
- `PyPDF2`
- `faiss`
- `python-dotenv`

### Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/conversational-study-assistant.git
   cd conversational-study-assistant
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Set Up Environmental Variables:** Create a `.env` file in the root of your project and add your Google API key:
   ```bash
   GOOGLE_API_KEY = your-google-api-key
4. **Run the Application:**
   ```bash
   streamlit run app.py

### How to Use
**1. PDF Q&A Bot:**
- **Upload PDFs:** Use the file uploader in the sidebar to upload your PDF files.
- **Ask Questions:** Once the PDFs are processed, type your question into the chat input, and the AI will provide an answer based on the content of the uploaded PDFs.
- **Clear History:** Use the sidebar button to clear the conversation history.
- **Download Conversation:** Option to download the conversation as a text file.

**2. Chatbot:** This works as a normal conventional chatbot built using Google's Gemini API

### Project Structure:
- `app.py`: The main application script.
- `requirements.txt`: List of all the dependencies needed for the project.
- `.env`: File to store environment variable like the Google API key.
- `faiss_index`: The directory where the FAISS vectore store is saved.

### Future Work:
Trying to work on adding few more features like analyzing various kind of file like `.txt`, `.docx`, `.csv` and etc.
