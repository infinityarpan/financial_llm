import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")
container = st.empty()

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
print(urls)
process_url_clicked=st.sidebar.button("Process URLs")

if process_url_clicked:
    document_loader=UnstructuredURLLoader(urls=urls)
    container.text("Document loader started........✅")
    documents=document_loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    container.text("Text splitter started........✅")
    texts=text_splitter.split_documents(documents)
    embeddings=OpenAIEmbeddings()
    container.text("Embedding and vector storage started........✅")
    faiss=FAISS.from_documents(texts, embeddings)
    faiss.save_local("vector_db")

query=container.text_input("question:")
if query:
    llm=OpenAI(temperature=0.6, max_tokens=500)
    embeddings=OpenAIEmbeddings()
    faiss=FAISS.load_local("vector_db", embeddings)
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss.as_retriever())
    result=chain({"question":query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])
    st.header("Sources")
    st.write(result["sources"])