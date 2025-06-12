# streamlit_app.py

import os
import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from query import query_similar
from respond import select_relevant_news, generate_response

# ——— Configuration ———
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "mlsum_tr_semantic")
EMBED_MODEL_ID = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = st.sidebar.number_input("Number of docs to retrieve (k)", min_value=1, max_value=20, value=10)
API_TIMEOUT = st.sidebar.number_input("Elasticsearch timeout (s)", min_value=1, max_value=60, value=10)

# ——— Initialize clients ———
@st.cache_resource(show_spinner=False)
def get_es_client():
    return Elasticsearch(ES_URL, verify_certs=True, timeout=API_TIMEOUT)

@st.cache_resource(show_spinner=False)
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL_ID)

es_client = get_es_client()
embed_model = get_embed_model()

# ——— Streamlit UI ———
st.title("📰 Turkish News RAG with LLaMA")
st.write(
    """
    Enter a query about Turkish news; the app will:
    1. Retrieve top-**k** relevant articles from Elasticsearch.  
    2. Select the 3 most relevant via LLaMA.  
    3. Generate a comprehensive answer based solely on those articles.
    """
)

prompt = st.text_area("Your query:", height=100)

if st.button("Generate Answer") and prompt.strip():
    with st.spinner("Retrieving documents…"):
        emb = embed_model.encode(prompt).tolist()
        hits = query_similar(emb, k=TOP_K, index=ES_INDEX, es=es_client)
        docs = [h["_source"] for h in hits]

    with st.expander(f"📄 Top {len(docs)} Retrieved Documents"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**[{i}] {doc.get('title','(no title)')}**")
            st.write(doc.get("summary", ""))
            st.write(doc.get("text","")[:300] + "…")
            st.markdown("---")

    with st.spinner("Selecting top 3…"):
        top3 = select_relevant_news(prompt, docs)

    with st.expander("🔍 Top 3 Selected for Answer"):
        for i, doc in enumerate(top3, 1):
            st.markdown(f"**[{i}] {doc.get('title','(no title)')}**")
            st.write(doc.get("summary", ""))
            st.write(doc.get("text","")[:300] + "…")
            st.markdown("---")

    with st.spinner("Generating answer…"):
        answer = generate_response(prompt, top3)

    st.subheader("💬 Generated Answer")
    st.write(answer)
