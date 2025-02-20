import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

milvus_token = os.getenv("MILVUS_TOKEN")

COLLECTION_NAME = "final_ragbench_document_embeddings"
MILVUS_CLOUD_URI = "https://in03-7b4da1b7b588a88.serverless.gcp-us-west1.cloud.zilliz.com"

#Function to create milvus db schema to insert the data
def CreateMilvusDbSchema():

    connections.connect("default", uri=MILVUS_CLOUD_URI, token=milvus_token)
    print(connections.get_connection_addr("default"))

    # Define the fields for the collection
    fields = [
        FieldSchema(name="chunk_doc_id", dtype=DataType.VARCHAR, max_length=350, is_primary=True, auto_id=False),  # Primary Key
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=300),  # Document ID
        FieldSchema(name="chunk_embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Vector Field (embedding)
        FieldSchema(name="context_relevance", dtype=DataType.FLOAT),  # Context Relevance Score
        FieldSchema(name="context_utilization", dtype=DataType.FLOAT),  # Context Utilization Score
        FieldSchema(name="adherence", dtype=DataType.FLOAT),  # Adherence Score
        FieldSchema(name="dataset_name", dtype=DataType.VARCHAR, max_length=300),  # Dataset Name
        FieldSchema(name="relevance_score", dtype=DataType.FLOAT),  # Relevance Score
        FieldSchema(name="utilization_score", dtype=DataType.FLOAT),  # Utilization Score
        FieldSchema(name="completeness_score", dtype=DataType.FLOAT)  # Completeness Score
    ]

    # Define the collection schema
    schema = CollectionSchema(fields, description="RAG Bench document vector collection")

    # Create the collection in Milvus
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Create an optimized index for fast vector search
    collection.create_index(
        "chunk_embedding",
        {
            "index_type": "HNSW",  # Hierarchical Navigable Small World (HNSW) index
            "metric_type": "COSINE",  # Cosine similarity for vector search
            "params": {"M": 16, "efConstruction": 200}  # HNSW parameters
        }
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully.")
    return collection