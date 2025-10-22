import streamlit as st
import os
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------------------------------------
# RAG BİLEŞENLERİNİN TANIMLANMASI
# -----------------------------------------------------------

try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"] 
except Exception as e:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY") 

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------------------------------------
# KRİTİK DÜZELTME: CHROMA DB'Yİ HER SEFERİNDE OLUŞTURMA
# -----------------------------------------------------------
CHROMA_DB_PATH = "./chroma_db"
PDF_PATH = "recipe book.pdf" 

if not os.path.exists(CHROMA_DB_PATH):

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Hocanın önerdiği boyut
    splits = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    st.success("Vektör veritabanı başarıyla oluşturuldu!")
else:

    vector_store = Chroma(
        embedding_function=embeddings, 
        persist_directory=CHROMA_DB_PATH
    )


retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=GEMINI_KEY)

prompt_template = """...""" 
prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------------------------------------
# STREAMLIT ARAYÜZ KODU
# -----------------------------------------------------------
st.title("🍰 Tarif Defteri Asistanı Chatbot (RAG)")
st.write("PDF'teki tariflere dayanarak sorularınızı cevaplıyorum.")

query = st.text_input("Hangi tatlının malzemelerini veya tarifini öğrenmek istersiniz?")

if query:
    with st.spinner('Tarif defterinde arama yapılıyor...'):
        response = rag_chain.invoke(query)
        st.success("Tarif Cevabı:")
        st.info(response)
