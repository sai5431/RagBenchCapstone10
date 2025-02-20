import pandas as pd
from datasets import load_dataset
from logger import logger
from typing import Dict, List


DATASET_CONFIGS = [
    'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa','tatqa', 'techqa'
]

#function to load the dataset for the given configurations.
#Args:configs: List of dataset configurations to load.
#Returns: A dictionary where keys are config names and values are the loaded datasets.
def load_rag_bench_dataset(configs: List[str]) -> Dict[str, dict]:

    ragbench = {}
    for config in configs:
        try:
            ragbench[config] = load_dataset("rungalileo/ragbench", config)
            logger.info(f"Successfully loaded dataset for config: {config}")
        except Exception as e:
            logger.error(f"Failed to load dataset for config {config}: {e}")
    return ragbench


#Extract data from the RAGBench dataset and store it in a Pandas DataFrame.
#Args:ragbench: Dictionary containing loaded datasets. split: Dataset split to extract (e.g., "train", "test", "validation").
#Returns:A Pandas DataFrame containing the extracted data.
def ExtractData(ragbench: Dict[str, dict], split: str = "train") -> pd.DataFrame:

    # Initialize a dictionary to store extracted data
    data = {
        "question": [],
        "documents": [],
        "gpt3_context_relevance": [],
        "gpt35_utilization": [],
        "gpt3_adherence": [],
        "id": [],
        "dataset_name": [],
        "relevance_score": [],
        "utilization_score": [],
        "completeness_score": [],
        "adherence_score": []

    }

    for datasetname, dataset in ragbench.items():
        try:
            # Ensure the split exists in the dataset
            if split not in dataset:
                logger.warning(f"Split '{split}' not found in dataset {datasetname}. Skipping.")
                continue

            # Extract data from the specified split
            split_data = dataset[split]

            # Check if required columns exist
            required_columns = ["question", "documents", "gpt3_context_relevance",
                                "gpt35_utilization", "gpt3_adherence", "id", "dataset_name"]
            missing_columns = [col for col in required_columns if col not in split_data.column_names]
            if missing_columns:
                logger.warning(f"Missing columns {missing_columns} in dataset {datasetname}. Skipping.")
                continue

            # Append data to lists
            data["question"].extend(split_data["question"])
            data["documents"].extend(split_data["documents"])
            data["gpt3_context_relevance"].extend(split_data["gpt3_context_relevance"])
            data["gpt35_utilization"].extend(split_data["gpt35_utilization"])
            data["gpt3_adherence"].extend(split_data["gpt3_adherence"])
            data["id"].extend(split_data["id"])
            data["dataset_name"].extend(split_data["dataset_name"])
            data["relevance_score"].extend(split_data["relevance_score"])
            data["utilization_score"].extend(split_data["utilization_score"])
            data["completeness_score"].extend(split_data["completeness_score"])
            data["adherence_score"].extend(split_data["adherence_score"])

            logger.info(f"Successfully extracted data from {datasetname} ({split} split).")
        except Exception as e:
            logger.error(f"Error extracting data from {datasetname} ({split} split): {e}")

    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)
    return df

def ExtractRagBenchData():
    ragbench = load_rag_bench_dataset(DATASET_CONFIGS)
    rag_extracted_data = ExtractData(ragbench, split="train")
    rag_extracted_data["dataset_name"].fillna("covidqa", inplace=True)

    return rag_extracted_data