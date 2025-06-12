from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
import os


def query_similar(embedding, k=5, index="mlsum_tr_semantic", host="localhost", port=9200, es=None):
    if not es:
        # connect with scheme
        es = Elasticsearch(f"http://{host}:{port}")

    # fetch embedding dim from index mapping
    mapping = es.indices.get_mapping(index=index)
    dims = mapping[index]["mappings"]["properties"]["embedding"]["dims"]
    if len(embedding) != dims:
        raise ValueError(
            f"Dimension mismatch: got {len(embedding)}, expected {dims}")

    # cosine similarity query (+1 to keep positive)
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }
    }
    return es.search(index=index, body=body)["hits"]["hits"]


def embed_prompt(prompt, model):
    return model.encode(prompt).tolist()


def print_similar(prompt, model, k=10, model_name="model"):
    try:
        results = query_similar(embed_prompt(prompt, model), k)
    except Exception as e:
        print("Error:", e)
        exit(1)

    # Print top-k results
    for hit in results:
        src = hit["_source"]
        print(prompt)
        print("=" * 80)
        print(f"ID={hit['_id']} Score={hit['_score']:.4f}")
        print(f"Title: {src.get('title')}")
        print(f"Topic: {src.get('topic')}")
        print(f"Summary: {src.get('summary')}")
        print(f"Text (truncated): {src.get('text', '')[:200]}...")
        print("=" * 80)

    # Save top-k results to CSV
    records = []
    for hit in results:
        src = hit["_source"]
        records.append({
            "prompt": prompt,
            "title": src.get("title"),
            "summary": src.get("summary"),
            "text": src.get("text", "")
        })

    df = pd.DataFrame(records)

    # Append to CSV
    file_exists = os.path.isfile("top_k_results.csv")
    df.to_csv("top_k_results.csv", mode='a', index=False,
              encoding="utf-8", header=not file_exists)

    print(f"Appended {len(df)} results to {model_name}_results.csv")
