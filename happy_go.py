import streamlit as st
import pandas as pd
from langchain_core.documents import Document 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader, TextLoader
import os
from tempfile import NamedTemporaryFile
import glob
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import email
import datetime

# UI Setup
st.set_page_config(page_title="🧠 Chat with Your Files", layout="wide")
st.title("📚 Ask Me Anything – Files, Folders & Emails")

# Session state for email memory and chat history
if 'gmail_docs' not in st.session_state:
    st.session_state.gmail_docs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - API Key and Options
openai_api_key = "sk-XBlDJqesAP5hLU0-gdsSpIeFVYETa4AsKKly-uTsVpT3BlbkFJ7cQzksch7r2a3U5HZ9RkANSMZuKmbvb-EBp1yLN-cA"
add_export = st.sidebar.checkbox("📤 Enable Export Chat to File")
if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.sidebar.success("Chat history cleared!")

# File Upload
uploaded_files = st.file_uploader("📎 Upload files (CSV, PDF, DOCX, PPTX, TXT)", type=["csv", "pdf", "docx", "pptx", "txt"], accept_multiple_files=True)

# Folder Upload Option
folder_path = st.text_input("📁 Or enter a folder path to load local files (including subfolders):")

# Gmail Auth and Load
if st.sidebar.button("📧 Load Emails from Gmail"):
    try:
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds = None

        if os.path.exists('token.pkl'):
            with open('token.pkl', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pkl', 'wb') as token:
                pickle.dump(creds, token)

        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=10, labelIds=['INBOX']).execute()
        messages = results.get('messages', [])

        gmail_docs = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            payload = msg_data['payload']
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '(Unknown Sender)')

            parts = payload.get('parts', [])
            body = ""
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break

            content = f"From: {sender}\nSubject: {subject}\n\n{body}"
            gmail_docs.append(Document(page_content=content, metadata={"source": "gmail"}))

        st.session_state.gmail_docs = gmail_docs
        st.sidebar.success("✅ Emails loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Gmail load failed: {e}")

if (uploaded_files or folder_path or st.session_state.gmail_docs) and openai_api_key:
    all_documents = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                if file_ext == "csv":
                    df = pd.read_csv(tmp_file_path, encoding='utf-8', engine='python')
                    documents = [
                        Document(page_content=row.to_string(index=False), metadata={"row": i, "source": uploaded_file.name})
                        for i, row in df.iterrows()
                    ]
                elif file_ext == "pdf":
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                elif file_ext == "docx":
                    loader = Docx2txtLoader(tmp_file_path)
                    documents = loader.load()
                elif file_ext == "pptx":
                    loader = UnstructuredPowerPointLoader(tmp_file_path)
                    documents = loader.load()
                elif file_ext == "txt":
                    loader = TextLoader(tmp_file_path)
                    documents = loader.load()
                else:
                    st.error(f"❌ Unsupported file type: {uploaded_file.name}")
                    documents = []
                all_documents.extend(documents)
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
            os.unlink(tmp_file_path)

    if folder_path:
        supported_exts = ["csv", "pdf", "docx", "pptx", "txt"]
        loaded_filepaths = []
        for filepath in glob.glob(os.path.join(folder_path, '**'), recursive=True):
            if os.path.isfile(filepath):
                ext = filepath.split(".")[-1].lower()
                if ext not in supported_exts:
                    continue
                try:
                    if ext == "csv":
                        df = pd.read_csv(filepath, encoding='utf-8', engine='python')
                        documents = [
                            Document(page_content=row.to_string(index=False), metadata={"row": i, "source": os.path.basename(filepath)})
                            for i, row in df.iterrows()
                        ]
                    elif ext == "pdf":
                        loader = PyPDFLoader(filepath)
                        documents = loader.load()
                    elif ext == "docx":
                        loader = Docx2txtLoader(filepath)
                        documents = loader.load()
                    elif ext == "pptx":
                        loader = UnstructuredPowerPointLoader(filepath)
                        documents = loader.load()
                    elif ext == "txt":
                        loader = TextLoader(filepath)
                        documents = loader.load()
                    all_documents.extend(documents)
                    loaded_filepaths.append(filepath)
                except Exception as e:
                    st.error(f"❌ Failed to process file {filepath}: {e}")

        if loaded_filepaths:
            with st.expander("📂 Loaded Files Summary"):
                st.markdown("**Total Files Loaded:** {}".format(len(loaded_filepaths)))
                for path in loaded_filepaths:
                    st.markdown(f"- `{path}`")

    all_documents.extend(st.session_state.gmail_docs)

    if all_documents:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(all_documents, embeddings)

        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        st.success("✅ Ready! Chat below with your content")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask something about your files or emails...")

        if user_query:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)

            with st.spinner("Thinking..."):
                result = qa_chain.run(user_query)

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(result)

            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": result})

        if add_export and st.button("💾 Export Chat to Text File"):
            chat_text = "\n\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history
            ])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(chat_text)
            with open(filename, "rb") as f:
                st.download_button("⬇️ Download Chat History", f, file_name=filename, mime="text/plain")
else:
    if not uploaded_files and not folder_path and not st.session_state.gmail_docs:
        st.warning("📎 Please upload files, enter a folder path, or load emails to get started.")
    elif not openai_api_key:
        st.warning("🔐 Please enter your OpenAI API Key to continue.")

