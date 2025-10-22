import streamlit as st
import os
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.colab import userdata

# -----------------------------------------------------------
# RAG Bileşenlerini Tanımlama (3. Aşamadan Tekrar Kullanım)
# -----------------------------------------------------------
try:
    GEMINI_KEY = userdata.get("GEMINI_API_KEY")
except:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY") # Diğer deploy platformları için

# Embedding Modelini Tanımla
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ChromaDB'yi Yükle (Dosyadan yükler, tekrar vektörleştirmez)
vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory="./chroma_db")
    name="langchain", # LangChain'in varsayılan adıdır
    embedding_function=embeddings
from langchain_community.vectorstores import Chroma
vector_store = Chroma(
    client=client,
    collection_name="langchain",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM Modelini Tanımla
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=GEMINI_KEY)

# Prompt Şablonu (Chatbot'un görevi)
prompt_template = """
Sen, sadece sağlanan tariflere dayanarak cevap veren bir Yemek Tarifi Uzmanısın.
Görev: Yalnızca aşağıda verilen bağlam (tarif) içinde sorulan soruya cevap ver.
Eğer tarif defterinde bilgi yoksa, "Bu tarif defterinde bu bilgi bulunmamaktadır." diye cevap ver.
Bağlam (Tarif): {context}
Soru: {question}
Cevap:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# LCEL Zincirini Kur
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------------------------------------
# STREAMLIT ARAYÜZÜ
# -----------------------------------------------------------
st.title("🍰 Tarif Defteri Asistanı Chatbot (RAG)")
st.write("PDF'teki tariflere (Vişneli Gül Tatlısı, Lor Tatlısı vb.) dayanarak sorularınızı cevaplıyorum.")

query = st.text_input("Hangi tatlının malzemelerini veya tarifini öğrenmek istersiniz?")

if query:
    with st.spinner('Tarif defterinde arama yapılıyor...'):
        response = rag_chain.invoke({"question": query})
        st.success("Tarif Cevabı:")
        st.info(response)
