import nltk
import pandas as pd
import numpy as np
nltk.data.path.append("/content/nltk_data")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize





#Splits a list of sentences into overlapping chunks using a sliding window approach.
#sentences (list): List of sentences to split into chunks.
#        window_size (int): Number of sentences in each chunk. Default is 6.
#        overlap (int): Number of overlapping sentences between consecutive chunks. Default is 3.
#    Returns:
#        list: List of text chunks, where each chunk is a string of concatenated sentences.

def split_into_sliding_windows(sentences, window_size=6, overlap=3):

    # Validate input parameters
    if window_size <= overlap:
        raise ValueError("window_size must be greater than overlap.")
    if not sentences:
        return []

    chunks = []
    step = window_size - overlap  # How much to move the window each time

    # Iterate over the sentences with the specified step size
    for i in range(0, len(sentences), step):
        chunk = sentences[i:i + window_size]
        if len(chunk) >= overlap:  # Ensure chunks have minimum required overlap
            chunks.append(" ".join(chunk))  # Join sentences into a text block

    return chunks

# Processes documents using a sliding window approach and inserts sentence chunks into Milvus.
#Args: model: The embedding model used to generate document embeddings.
#   extracted_data: Pandas DataFrame containing the extracted data.
#    collectionInstance: Milvus collection instance to insert data into.
#    window_size: Number of sentences in each chunk.
#    overlap: Number of overlapping sentences between consecutive chunks.
#

def EmbedAllDocumentsAndInsert(model, extracted_data, collectionInstance, window_size=5, overlap=2):

    count = 0
    total_docs = len(extracted_data)
    print(f"Total documents: {total_docs}")

    for index, row in extracted_data.iterrows():
        document = row["documents"]  # Extract the document text
        doc_id = row["id"]  # Extract the document ID
        doccontextrel = row["gpt3_context_relevance"]  # Extract context relevance score
        doccontextutil = row["gpt35_utilization"]  # Extract context utilization score
        docadherence = row["gpt3_adherence"]  # Extract adherence score
        datasetname = row["dataset_name"]  # Extract dataset name
        relevance_score = row["relevance_score"]  # Extract relevance score
        utilization_score = row["utilization_score"]  # Extract utilization score
        completeness_score = row["completeness_score"]  # Extract completeness score


        if isinstance(document, list):
            # Flatten the list into a single string
            document = " ".join([str(item) for item in document if isinstance(item, str)])
        elif not isinstance(document, str):
            # If the document is not a string or list, convert it to a string
            document = str(document)

        # Step 1: Tokenize document into sentences
        sentences = sent_tokenize(document) if isinstance(document, str) else document

        # Step 2: Generate overlapping chunks
        chunks = split_into_sliding_windows(sentences, window_size, overlap)

        print(f"Total chunks for document {index}: {len(chunks)}")

        for chunk_index, chunk_text in enumerate(chunks):
            # Step 3: Generate embedding for each chunk
            chunk_vector = np.array(model.encode(chunk_text), dtype=np.float32).flatten().tolist()

            print(f"chunk_index= {chunk_index}")

            # Step 4: Insert chunk into Milvus as separate columns
            insert_embeddings_into_milvus(
                collectionInstance,
                chunk_vector,
                f"{chunk_index}__{doc_id}",  # Unique ID for chunk
                doc_id,  # Unique ID for doc
                index,
                float(doccontextrel) if pd.notna(doccontextrel) else 0.0,  # Handle NaN values
                float(doccontextutil) if pd.notna(doccontextutil) else 0.0,  # Handle NaN values
                float(docadherence) if pd.notna(docadherence) else 0.0,  # Handle NaN values
                datasetname,  # Dataset name column
                float(relevance_score) if pd.notna(relevance_score) else 0.0,  # Handle NaN values
                float(utilization_score) if pd.notna(utilization_score) else 0.0,  # Handle NaN values
                float(completeness_score) if pd.notna(completeness_score) else 0.0  # Handle NaN values
            )

            count += 1
            if count % 1000 == 0:
                print(f"Uploaded {count} chunks to Milvus.")

# Inserts document embeddings into Milvus along with metadata.
#Args:
#        collection: Milvus collection instance.
#        embeddings: Embedding vector for the chunk.
#        chunk_doc_id: Unique ID for the chunk.
#        doc_id: Unique ID for the document.
#       index: Index of the document in the dataset.
#        doccontextrel: Context relevance score.
#        doccontextutil: Context utilization score.
#       docadherence: Adherence score.
#       datasetname: Name of the dataset.

def insert_embeddings_into_milvus(collection, embeddings, chunk_doc_id, doc_id, index,
                                  doccontextrel, doccontextutil, docadherence, datasetname,
                                  relevance_score, utilization_score, completeness_score):

    try:
        print(f"Inserting chunk {chunk_doc_id} doc {doc_id} (index {index})")
        insert_data = [
            [str(chunk_doc_id)],  # Primary key field (document_id)
            [str(doc_id)],  # Document ID field
            [embeddings],  # Vector field (embedding)
            [float(doccontextrel)],  # Relevance score field
            [float(doccontextutil)],  # Utilization score field
            [float(docadherence)],  # Adherence score field
            [str(datasetname)],  # Dataset name field
            [float(relevance_score)],  # Relevance score field
            [float(utilization_score)],  # Utilization score field
            [float(completeness_score)]  # Completeness score field
        ]
        collection.insert(insert_data)
    except Exception as e:
        print(f"Error inserting chunk {chunk_doc_id} doc {doc_id} (index {index}): {e}")                


