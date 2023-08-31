import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


openai.api_type = "azure"
openai.api_base = "https://meus-aai-dev-voxelgenerate-001.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = ""

def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(
                engine = "voxelgenerate-gpt-35-turbo",
                openai_api_base = openai.api_base,
                openai_api_type = openai.api_type, 
                openai_api_key = openai.api_key
        )
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key = openai.api_key, 
                                                    engine="voxelgenerate-gpt-35-turbo"),
                                                    chain_type='stuff', retriever=retriever)
        return qa.run(query_text)


# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)
