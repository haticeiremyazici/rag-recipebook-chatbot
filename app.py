# -----------------------------------------------------------
# APP.PY NİHAİ VE HATASIZ VERSİYON
# -----------------------------------------------------------
import streamlit as st
import os
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection # <<< YENİ BAĞLANTI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # <<< CHROMA MODÜLÜNÜN STABİL YOLU
from streamlit import secrets 

# -----------------------------------------------------------
# 1. BAĞLANTI VE RAG SİSTEMİ KURULUMU
# -----------------------------------------------------------

# API Key'i Streamlit Secrets'tan okuma
try:
    GEMINI_KEY = secrets["GEMINI_API_KEY"]
except:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY") 

# Embedding Modeli (HuggingFace, kota sorununu aşan model)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ChromaDB'ye bağlanma (st.connection kullanılarak)
# path: Colab'de oluşturduğumuz ve GitHub'a yüklediğimiz klasörün yolu.
conn = st.connection("chromadb", type=ChromadbConnection, path="./chroma_db", embedding_function=embeddings)

# Chroma'dan LangChain Retriever'ı çekme
# Not: st.connection'ın döndürdüğü client'ı LangChain'e tanıtıyoruz.
vector_store = Chroma(client=conn.get_client(), collection_name="langchain", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM Modelini Tanımla
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=GEMINI_KEY)

# Prompt ve LCEL Zinciri (Aynı kalır)
prompt_template = """
Sen, sadece sağlanan tariflere dayanarak cevap veren bir Yemek Tarifi Uzmanısın. [...]
Bağlam (Tarif): {context}
Soru: {question}
Cevap:
"""
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
st.write("PDF'teki tariflere dayanarak cevap veriyorum.")

query = st.text_input("Hangi tatlının malzemelerini veya tarifini öğrenmek istersiniz?")

if query:
    with st.spinner('Tarif defterinde arama yapılıyor...'):
        response = rag_chain.invoke(query)
        st.success("Tarif Cevabı:")
        st.info(response)
