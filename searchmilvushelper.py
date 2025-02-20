#Search Milvus by generating an embedding for the query text. Returns the top_k most similar documents.
#Retrieves all columns defined in the Milvus schema.

def SearchTopKDocuments(collection, query_text, model, top_k=10):

    # Generate embedding for the query text
    query_embedding = model.encode(query_text, convert_to_numpy=True)

    # Define search parameters
    search_params = {
        "metric_type": "COSINE",  # Similarity metric
        "params": {"ef": 64}      # Controls recall, higher values = better accuracy but slower
    }

    # Perform the search
    results = collection.search(
        data=[query_embedding],
        anns_field="chunk_embedding",  # Field containing the embeddings
        param=search_params,
        limit=top_k,
        output_fields=[
            "chunk_doc_id",  # Primary key
            "doc_id",        # Document ID
            "context_relevance",  # Context Relevance Score
            "context_utilization",  # Context Utilization Score
            "adherence",  # Adherence Score
            "dataset_name",  # Dataset Name
            "relevance_score",  # Relevance Score
            "utilization_score",  # Utilization Score
            "completeness_score"  # Completeness Score
        ]
    )

    # Process and return the results
    top_documents = []
    for hits in results:
        for hit in hits:
            doc = {
                "chunk_doc_id": hit.entity.get("chunk_doc_id"),  # Primary key
                "doc_id": hit.entity.get("doc_id"),  # Document ID
                "context_relevance": hit.entity.get("context_relevance"),  # Context Relevance Score
                "context_utilization": hit.entity.get("context_utilization"),  # Context Utilization Score
                "adherence": hit.entity.get("adherence"),  # Adherence Score
                "dataset_name": hit.entity.get("dataset_name"),  # Dataset Name
                "relevance_score": hit.entity.get("relevance_score"),  # Relevance Score
                "utilization_score": hit.entity.get("utilization_score"),  # Utilization Score
                "completeness_score": hit.entity.get("completeness_score"),  # Completeness Score
                "distance": hit.distance  # Similarity score (cosine distance)
            }
            top_documents.append(doc)

    return top_documents