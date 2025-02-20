import gradio as gr
import os
import time

from loaddataset import ExtractRagBenchData
from createmilvusschema import CreateMilvusDbSchema
from insertmilvushelper import EmbedAllDocumentsAndInsert
from sentence_transformers import SentenceTransformer
from searchmilvushelper import SearchTopKDocuments
from finetuneresults import FineTuneAndRerankSearchResults
from generationhelper import GenerateAnswer
from formatresultshelper import FormatAndScores
from calculatescores import CalculateScores

from huggingface_hub import login
from huggingface_hub import whoami
from huggingface_hub import dataset_info


# Load embedding model
QUERY_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
RERANKING_MODELS = {
    "MS MARCO MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MonoT5 Base": "castorini/monot5-base-msmarco",
}
PROMPT_MODEL = "llama-3.3-70b-specdec"
EVAL_MODEL = "llama-3.3-70b-specdec"
WINDOW_SIZE = 5
OVERLAP = 2
RETRIVE_TOP_K_SIZE=10


hf_token = os.getenv("HF_TOKEN")
login(hf_token)

rag_extracted_data = ExtractRagBenchData()
print(rag_extracted_data.head(5))

"""
EmbedAllDocumentsAndInsert(QUERY_EMBEDDING_MODEL, rag_extracted_data, db_collection, window_size=WINDOW_SIZE, overlap=OVERLAP)
"""  

def EvaluateRAGModel(question, evaluation_model, reranking_model):
    try:
        start_time = time.time()

        query = question.strip()

        if evaluation_model == "LLaMA 3.3":
            EVAL_MODEL = "llama-3.3-70b-specdec"
            PROMPT_MODEL = "llama-3.3-70b-specdec"
        elif evaluation_model == "Mistral 7B":
            EVAL_MODEL = "mixtral-8x7b-32768"
            PROMPT_MODEL = "mixtral-8x7b-32768"
        elif evaluation_model == "Deepseek 70b":
            EVAL_MODEL = "deepseek-r1-distill-llama-70b"
            PROMPT_MODEL = "deepseek-r1-distill-llama-70b"
        
        # Get selected reranking model
        RERANKING_MODEL = RERANKING_MODELS[reranking_model]
        
        #invoke create milvus db function
        try:
            db_collection = CreateMilvusDbSchema()
        except Exception as e:
            print(f"Error creating Milvus DB schema: {e}")

        #insert embdeding to milvus db

        #query = "what would the net revenue have been in 2015 if there wasn't a stipulated settlement from the business combination in october 2015?"

        results_for_top10_chunks = SearchTopKDocuments(db_collection, query, QUERY_EMBEDDING_MODEL, top_k=RETRIVE_TOP_K_SIZE)

        reranked_results = FineTuneAndRerankSearchResults(results_for_top10_chunks, rag_extracted_data, query, RERANKING_MODEL)

        answer = GenerateAnswer(query, reranked_results.head(3), PROMPT_MODEL)

        completion_result,relevant_sentence_keys,all_utilized_sentence_keys,support_keys,support_level = FormatAndScores(query, reranked_results.head(1), answer, EVAL_MODEL)


        print(relevant_sentence_keys)
        print(all_utilized_sentence_keys)
        print(support_keys)
        print(support_level)
        print(completion_result)

        document_id = reranked_results.head(1)['doc_id'].values[0]
        extarcted_row_for_given_id = rag_extracted_data[rag_extracted_data["id"]==document_id]

        rmsecontextrel, rmsecontextutil, aucscore = CalculateScores(relevant_sentence_keys,all_utilized_sentence_keys,support_keys,support_level,extarcted_row_for_given_id)

        print(rmsecontextrel)
        print(rmsecontextutil)
        print(aucscore)
        end_time = time.time()

        execution_time = end_time - start_time

        return answer, rmsecontextrel, rmsecontextutil, aucscore, execution_time, gr.update(visible=False)
    except Exception as e:
        error_message = f"""
        <div style="background-color: #ffcccc; color: red; padding: 10px; border-radius: 5px; font-weight: bold;">
            ⚠️ <b>Error:</b> {str(e)}
        </div>
        """
        return "I apologize, but I encountered an error processing your question. Please try again.", 0, 0, 0, time.time() - start_time, gr.update(value=error_message, visible=True)

# Create Gradio UI
with gr.Blocks() as iface:
    gr.Markdown("## Capstone Project Group 10 ")
    

    with gr.Row():
        question_input = gr.Textbox(label="Enter your Question", lines=5)
        with gr.Column(scale=0.5):

            dropdown_input = gr.Dropdown(
                ["LLaMA 3.3", "Mistral 7B","Deepseek 70b"], 
                value="LLaMA 3.3", 
                label="Select a Model"
            )

            reranker_dropdown = gr.Dropdown(
                list(RERANKING_MODELS.keys()),
                value="MS MARCO MiniLM",
                label="Select Reranking Model"
            )

    submit_button = gr.Button("Evaluate Model")

    # Simulated "Popup" Error Message (Initially Hidden)
    error_message_box = gr.HTML("", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Response")
            response = gr.Textbox(interactive=False, show_label=False, lines=2)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### RMSE-CONTEXT RELEVANCE")
            rmsecontextrel = gr.Textbox(interactive=False, show_label=False, lines=2)
    
        with gr.Column():
            gr.Markdown("### RMSE-CONTEXT UTILIZATION")
            rmsecontextutil = gr.Textbox(interactive=False, show_label=False, lines=2)

        with gr.Column():
            gr.Markdown("### AUCROC ADHERENCE")
            aucscore = gr.Textbox(interactive=False, show_label=False, lines=2)

        with gr.Column():
            gr.Markdown("### PROCESSING TIME")
            processingTime = gr.Textbox(interactive=False, show_label=False, lines=2)


    # Connect submit button to evaluation function
    submit_button.click(
        EvaluateRAGModel, 
        inputs=[question_input, dropdown_input,reranker_dropdown], 
        outputs=[response, rmsecontextrel, rmsecontextutil, aucscore, processingTime, error_message_box]
        )

# Run the Gradio app
iface.launch()