# Install dependencies first
# pip install faiss-cpu sentence-transformers streamlit pandas

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

# -------------------------
# 1️⃣ Load FAQ data
# -------------------------
# You can also load from a CSV file with columns: question, answer
faq_data = [
    {"question": "How can I reset my password?", "answer": "Click on 'Forgot password' to reset it."},
    {"question": "What is your refund policy?", "answer": "You can request a refund within 30 days."},
    {"question": "How to contact support?", "answer": "Email us at support@example.com."},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship worldwide."},
    {"question": "Do operate internationally?", "answer": "Yes, we operate internationally."},
]

df = pd.DataFrame(faq_data)

# -------------------------
# 2️⃣ Initialize model & create embeddings
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, small model
faq_embeddings = model.encode(df['question'].tolist(), convert_to_numpy=True)

# -------------------------
# 3️⃣ Build FAISS index
# -------------------------
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(faq_embeddings)

# -------------------------
# 4️⃣ Query function
# -------------------------
def query_faq(user_question, top_k=1, threshold=0.5):
    user_emb = model.encode([user_question], convert_to_numpy=True)
    distances, indices = index.search(user_emb, top_k)
    
    # Convert L2 distance to similarity score
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 / (1 + dist)  # simple conversion: higher = more similar
        if similarity >= threshold:
            results.append({
                "question": df.iloc[idx]['question'],
                "answer": df.iloc[idx]['answer'],
                "similarity": float(similarity)
            })
    if not results:
        return [{"question": None, "answer": "Sorry, I couldn't find a good match.", "similarity": 0}]
    return results

# -------------------------
# 5️⃣ Streamlit Interface
# -------------------------
st.title("🧠 FAQ Bot (Retrieval-Augmented)")
user_input = st.text_input("Ask your question:")

if user_input:
    answers = query_faq(user_input, top_k=3, threshold=0.5)
    for ans in answers:
        st.write(f"**Matched Question:** {ans['question']}")
        st.write(f"**Answer:** {ans['answer']}")
        st.write(f"*Similarity Score:* {ans['similarity']:.2f}")
        st.markdown("---")