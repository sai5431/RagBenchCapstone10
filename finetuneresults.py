from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import List, Tuple

class MonoT5Reranker:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, query_doc_pairs: List[Tuple[str, str]]) -> np.ndarray:
        scores = []
        batch_size = 8  # Adjust based on your GPU/CPU memory
        
        for i in range(0, len(query_doc_pairs), batch_size):
            batch_pairs = query_doc_pairs[i:i + batch_size]
            
            # Format input as per MonoT5 requirements
            inputs = [f"Query: {query} Document: {doc}" for query, doc in batch_pairs]
            
            # Tokenize
            encoded = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        return np.array(scores)

class MSMARCOReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        
    def predict(self, query_doc_pairs: List[Tuple[str, str]]) -> np.ndarray:
        return self.model.predict(query_doc_pairs)


def get_reranker(model_name: str):
    """Factory function to get appropriate reranker based on model name."""
    if "monot5" in model_name.lower():
        print(f"Using MonoT5 reranker: {model_name}")
        return MonoT5Reranker(model_name)
    else:
        print(f"Using MS MARCO reranker: {model_name}")
        return MSMARCOReranker(model_name)
    
"""
    Retrieves unique full documents based on the top-ranked document IDs.

    Args:
        top_documents (list): List of dictionaries containing 'doc_id'.
        df (pd.DataFrame): The dataset containing document IDs and text.

    Returns:
        pd.DataFrame: A DataFrame with 'doc_id' and 'document'.
"""
def retrieve_full_documents(top_documents, df):

    # Extract unique doc_ids
    unique_doc_ids = list(set(doc["doc_id"] for doc in top_documents))

    # Print for debugging
    print(f"Extracted Doc IDs: {unique_doc_ids}")

    # Filter DataFrame where 'id' matches any of the unique_doc_ids
    filtered_df = df[df["id"].isin(unique_doc_ids)][["id", "documents"]].drop_duplicates(subset="id")

    # Rename columns for clarity
    filtered_df = filtered_df.rename(columns={"id": "doc_id", "documents": "document"})

    return filtered_df

"""
Reranks the retrieved documents based on their relevance to the query using a Cross-Encoder model.
Args:
     query (str): The search query.
     retrieved_docs (pd.DataFrame): DataFrame with 'doc_id' and 'document'.
     model_name (str): Name of the Cross-Encoder model.
Returns:
     pd.DataFrame: A sorted DataFrame with doc_id, document, and reranking score.
"""

def rerank_documents(query, retrieved_docs_df, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Reranks documents using the specified reranking model."""
    try:
        # Load Cross-Encoder model
        model = get_reranker(model_name)

        # Prepare query-document pairs
        query_doc_pairs = [(query, " ".join(doc)) for doc in retrieved_docs_df["document"]]

        # Compute relevance scores
        scores = model.predict(query_doc_pairs)

        # Add scores to the DataFrame
        retrieved_docs_df["relevance_score"] = scores

        # Sort by score in descending order (higher score = more relevant)
        reranked_docs_df = retrieved_docs_df.sort_values(by="relevance_score", ascending=False).reset_index(drop=True)

        return reranked_docs_df
    except Exception as e:
            print(f"Error in reranking: {e}")
            # Return original order if reranking fails
            retrieved_docs_df["relevance_score"] = 1.0
            return retrieved_docs_df

def FineTuneAndRerankSearchResults(top_10_chunk_results, rag_extarcted_data, question, reranking_model):
    try:
        unique_docs= retrieve_full_documents(top_10_chunk_results, rag_extarcted_data)
        reranked_results = rerank_documents(question, unique_docs, reranking_model)
        return reranked_results
    except Exception as e:
        print(f"Error in FineTuneAndRerankSearchResults: {e}")
        return None