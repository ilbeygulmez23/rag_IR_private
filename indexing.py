from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def index_data(model):

    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")

    # Define the index name
    index_name = "mlsum_tr_semantic"

    # Delete the index if it already exists
    if es.indices.exists(index=index_name, request_timeout=30):
        print("Data is already indexed")
    else:
        # Create the index with appropriate mappings
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "summary": {"type": "text"},
                        "title": {"type": "text"},
                        "topic": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": model.get_sentence_embedding_dimension()
                        }
                    }
                }
            },
            request_timeout=60
        )

        # Load the Turkish portion of the MLSUM dataset
        dataset = load_dataset(
            "mlsum", "tu", split="train[:5%]", trust_remote_code=True)

        # Initialize the embedding model
        # model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generate embeddings for each text entry
        dataset = dataset.map(
            lambda batch: {"embedding": model.encode(batch["text"]).tolist()})

        # Prepare documents for bulk indexing
        documents = [
            {
                "_index": index_name,
                "_id": idx,
                "_source": {
                    "text": row["text"],
                    "summary": row["summary"],
                    "title": row["title"],
                    "topic": row["topic"],
                    "embedding": row["embedding"]
                }
            }
            for idx, row in enumerate(dataset)
        ]

        # Bulk index the documents
        bulk(es, documents)
