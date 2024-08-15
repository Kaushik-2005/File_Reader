import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Conversational Study Assistant",
    page_icon="📚",
    layout="centered",
)

# Custom CSS
st.markdown("""
            <style>
                .sidebar .sidebar-content{
                    padding-top: 2rem;
                }
                .stRadio > div {
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    background-color: #2c3f33;
                    padding: 10px;
                    border-radius: 8px;
                }
                .stRadio > div label{
                    font-size: 1rem;
                    font-weight: bold;
                    color: white;
                }
                .stButton button{
                    background-color: #7289da;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    font-size: 1rem;
                    width: 100%;
                    cursor: pointer;
                    margin-bottom: 10px;
                }
                .stButton button:hover{
                    background-color: #5b6eae;
                }
                .stDownloadButton button:hover {
                    background-color: #7e8b99;
                }
                .css-1l02zno {
                    padding: 1rem;
                }
                .css-1aumxhk {
                    padding-top: 1rem;
                }
                .stFileUploader label {
                    font-weight: bold;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

# Function to create a conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "The answer is not available in the context," and don't provide a wrong answer.
    Context: \n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

# Function to handle user input and generate a response
def handle_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Keep track of the conversation history
    conversation_history = st.session_state.get('conversation_history', [])

    # Process the current user question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Append the current question and answer to the history
    conversation_history.append((user_question, response['output_text']))
    st.session_state['conversation_history'] = conversation_history

    # Display the conversation history
    for question, answer in conversation_history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {question}")
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {answer}")

# Function to clear the conversation history
def clear_conversation():
    st.session_state['conversation_history'] = []  # Clear history
    st.session_state.chat_session = genai.GenerativeModel('gemini-pro').start_chat(history=[])  # Start a new chat session
    st.experimental_rerun()  # Rerun the app to refresh the state

# Chatbot function similar to bot.py
def start_chatbot():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = genai.GenerativeModel('gemini-pro').start_chat(history=[])

    def translate_role_for_streamlit(user_role):
        return "assistant" if user_role == "model" else user_role

    st.title("Gemini-Chatbot")

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append(f"You: {user_prompt}")

        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        st.session_state.chat_history.append(f"Assistant: {gemini_response.text}")

        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Main function to run the app
def main():
    # Sidebar for navigation
    with st.sidebar:
        st.title("Menu")

        # Select between Study Assistant and Chatbot
        mode = st.radio("Choose Mode", ("PDF qna bot", "Chatbot"))

        # Clear conversation history button
        st.button('Clear Conversation History', on_click=clear_conversation)

        # Option to download the conversation history
        if 'conversation_history' in st.session_state:
            chat_history_str = "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state['conversation_history']])
            st.download_button(
                label="Download chat",
                data=chat_history_str,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    # Conditional content based on the mode selected
    if mode == "Chatbot":
        start_chatbot()
    elif mode == "PDF qna bot":
        st.title("PDF qna Assistant")

        # Sidebar options specific to Study Assistant
        with st.sidebar:
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete.")
                else:
                    st.error("Please upload at least one PDF file.")

        st.subheader("Ask a Question")
        user_question = st.chat_input("Type your question here...")
        if user_question:
            handle_user_input(user_question)

if __name__ == "__main__":
    main()
