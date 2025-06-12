# app.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

from query import query_similar
from respond import select_relevant_news, generate_response  # your updated respond.py

# — optional: read these from env or hard-code —
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX = os.getenv("ES_INDEX", "mlsum_tr_semantic")
EMBED_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)
ES_URL = os.getenv("ES_URL", "http://localhost:9200")

# ——— Logging configuration ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize once
logger.info("Initializing Elasticsearch client at %s", ES_URL)
es_client = Elasticsearch(ES_URL)

logger.info("Loading embedding model %s", EMBED_MODEL_ID)
embed_model = SentenceTransformer(EMBED_MODEL_ID)

app = FastAPI(title="RAG with Turkish LLaMA", version="1.0")


class PromptRequest(BaseModel):
    prompt: str


class Doc(BaseModel):
    title: str
    summary: str
    text: str


class RAGResponse(BaseModel):
    prompt: str
    selected_docs: list[Doc]
    answer: str


@app.post("/respond", response_model=RAGResponse)
def respond_endpoint(req: PromptRequest):
    try:
        logger.info("Received prompt: %s", req.prompt)

        # 1) embed & retrieve
        emb = embed_model.encode(req.prompt).tolist()
        logger.info("Generated embedding of length %d", len(emb))

        hits = query_similar(emb, k=10, index=ES_INDEX, es=es_client)
        logger.info("Retrieved %d documents from Elasticsearch", len(hits))

        docs = [h["_source"] for h in hits]

        # 2) Select top 3
        top3 = select_relevant_news(req.prompt, docs)
        logger.info("Selected top %d relevant docs", len(top3))

        # 3) Generate final answer
        ans = generate_response(req.prompt, top3)
        logger.info("Generated answer of length %d", len(ans))

        return RAGResponse(
            prompt=req.prompt,
            selected_docs=[Doc(**d) for d in top3],
            answer=ans
        )

    except Exception as e:
        logger.exception("Error in /respond endpoint")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on 0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
