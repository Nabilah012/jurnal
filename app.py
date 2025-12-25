import streamlit as st
import pickle
import re

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("model_nb.pkl", "rb") as f:
        nb, tfidf, le = pickle.load(f)
    return nb, tfidf, le

model, tfidf, le = load_model()

# =========================
# PREPROCESSING (SAMA DENGAN COLAB)
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# =========================
# STREAMLIT UI
# =========================
st.title("Analisis Sentimen Review Produk")
st.write("Metode: **TF-IDF + Naive Bayes**")

review_input = st.text_area("Masukkan review produk:")

if st.button("Prediksi Sentimen"):
    if review_input.strip() == "":
        st.warning("Review tidak boleh kosong!")
    else:
        clean_text = preprocess_text(review_input)
        X_input = tfidf.transform([clean_text])
        prediction = model.predict(X_input)
        label = le.inverse_transform(prediction)[0]

        st.success(f"Hasil Prediksi Sentimen: **{label}**")
