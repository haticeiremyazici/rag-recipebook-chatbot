# 🍰 RAG (Retrieval Augmented Generation) Chatbot Projesi: Tarif Defteri Asistanı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş RAG temelli bir uygulamadır. Amacı, sağlanan el yazması tarif defterine dayanarak kullanıcı sorularına hızlı, bağlamsal ve doğru cevaplar vermektir.

---

## 🎯 1. PROJENİN AMACI VE ELDE EDİLEN SONUÇLAR

Bu chatbot, el yazması PDF formatındaki tarif defterinde yer alan bilgilere dayanarak, kullanıcıların tariflerle ilgili spesifik sorgularını yanıtlamayı amaçlar. Model, genel bilgi vermek yerine, **yalnızca defterdeki bilgiyle** cevap üretir.

**Elde Edilen Sonuç:** Sistem, karşılaşılan tüm teknik zorluklara rağmen, modern LCEL mimarisi ile kurulmuş ve Lor Tatlısı gibi spesifik sorgulara dahi **doğru ve bağlamsal** cevap verebilir durumdadır.

## 💾 2. VERİ SETİ HAKKINDA BİLGİ

* **Veri Kaynağı:** `recipe book.pdf` dosyası.
* **İçerik:** Vişneli Gül Tatlısı, Lor Tatlısı, Baklava ve çeşitli Kurabiye/Kekler dahil olmak üzere **30'dan fazla tatlı ve tuzlu tarifin** malzeme listesi ve hazırlanış adımları.
* **Veri Tipi:** Metin tabanlı (PDF).

## ⚙️ 3. ÇÖZÜM MİMARİSİ (RAG PİPELİNE)

Proje, LangChain'in modern **LCEL (LangChain Expression Language)** yöntemi kullanılarak oluşturulan bir RAG zincirine dayanmaktadır.

| Bileşen | Seçim | Rolü ve Açıklaması |
| :--- | :--- | :--- |
| **RAG Çerçevesi** | LCEL (LangChain Expression Language) | Kompleks RAG zincirini hatasız ve stabil bir şekilde çalıştırmıştır. |
| **Büyük Dil Modeli (LLM)** | Gemini 2.5 Flash API | Cevap üretir (Generation). |
| **Embedding Modeli** | HuggingFace `all-MiniLM-L6-v2` | Metinleri vektörlere çevirmiştir. (Gemini **kota aşımlarını** atlamak için açık kaynaklı modele geçilmiştir.) |
| **Vektör Veritabanı (Vector DB)** | ChromaDB | Vektörleri depolar ve sorguya en yakın **tarif parçalarını** çeker (Retrieval). |

## 🛠️ 4. KODUN ÇALIŞMA KILAVUZU

Bu proje Google Colab ortamında geliştirilmiştir.

1.  **Dosya Hazırlığı:** `app.py`, `requirements.txt` ve `recipe book.pdf` dosyaları GitHub'dan indirilerek Colab ortamına yüklenmelidir.
2.  **API Anahtarları:** Colab **Secrets** üzerinden **`GEMINI_API_KEY`** anahtarları eklenmelidir.
3.  **Kurulum:** Colab hücresinde `!pip install -r requirements.txt` komutu çalıştırılmalıdır.
4.  **RAG Kurulumu:** Colab notebook'undaki tüm RAG kurulum (Embedding ve Zincir) kodları çalıştırılmalıdır.
5.  **Arayüzü Başlatma:** Streamlit aşağıdaki komutlar ile Ngrok üzerinden tünellenerek canlıya alınır:
    ```bash
    !ngrok authtoken [NGROK ANAHTARINIZ]
    !streamlit run app.py & npx kill-port 8501
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    ```

## 🌐 5. WEB ARAYÜZÜ & PRODUCT KILAVUZU (CANLI DEPLOY)

Proje, **Streamlit** kullanılarak web arayüzü üzerinden sunulmaktadır.

**CANLI DEPLOY LİNKİ:** https://rag-recipebook-chatbot.streamlit.app/
**Test Senaryosu:**

Kullanıcı, yukarıdaki linke giderek, tarif defterinden bir soru sorar. Örneğin: **"Lor Tatlısı'nın şerbeti için kaç bardak su ve şeker gereklidir?"**
* **Beklenen Cevap:** Sistem, PDF'i tarar ve Lor Tatlısı'na ait bilgiyi çekerek cevap verir.

NOTLAR:
Blocklanma için streamlit cloudda deploy edilen repoda çok değişiklik yapılması hata vermesine neden olabiliyormuş. "Your account has  exceeded fair-use limits and was blocked by the system." bu uyarıyla streamlit hesabıma da tekrar giremediğim için yeni deploy da oluşturamadım. Mentorum yaşadığım durumu READ.ME dosyama eklememi söyledi.
