---
title: RagBenchCapstone10
emoji: 📉
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
short_description: RagBench Dataset development by Saiteja
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# RAG Benchmark Evaluation System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for evaluating different language models and reranking strategies. It provides a user-friendly interface for querying documents and analyzing the performance of various models.

## Features

- Multiple LLM support (LLaMA 3.3, Mistral 7B)
- Various reranking models:
  - MS MARCO MiniLM
  - MS MARCO TinyBERT
  - MonoT5 Base
  - MonoT5 Small
  - MonoT5 3B
- Vector similarity search using Milvus
- Automatic document chunking and retrieval
- Performance metrics calculation
- Interactive Gradio interface

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   bash
   git clone https://github.com/yourusername/rag-benchmark.git
   cd rag-benchmark

2. Install dependencies:

- pip install -r requirements.txt

3. Configure the models:

- Create a `models` directory and add your language model files.
- Create a `rerankers` directory and add your reranking model files.

- Run the application:

- python app.py

## Usage

1. Start the application:

2. Access the web interface at `http://localhost:7860`

3. Enter your question and select:

   - LLM Model (LLaMA 3.3 or Mistral 7B)
   - Reranking Model (MS MARCO or MonoT5 variants)

4. Click "Evaluate Model" to get results

## Metrics

The system calculates several performance metrics:

- RMSE Context Relevance
- RMSE Context Utilization
- AUCROC Adherence
- Processing Time

## Reranking Models Comparison

### MS MARCO Models

- **MiniLM**: Fast and efficient, good general performance
- **TinyBERT**: Lightweight, slightly lower accuracy but faster

### MonoT5 Models

- **Small**: Compact and fast, suitable for limited resources
- **Base**: Balanced performance and speed
- **3B**: Highest accuracy, requires more computational resources

## Error Handling

- Automatic fallback to fewer documents if token limits are exceeded
- Graceful handling of API timeouts
- Comprehensive error logging

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Dependencies

- gradio
- torch
- transformers
- sentence-transformers
- pymilvus
- numpy
- pandas
- scikit-learn
- tiktoken
- groq
- huggingface_hub

## License

[Your License Here]

## Acknowledgments

- RAGBench dataset
- Hugging Face Transformers
- Milvus Vector Database
- Groq API
