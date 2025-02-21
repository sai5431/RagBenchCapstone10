import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_preprocess_data(file_path):
    # Read Excel file, skipping the first 2 rows
    df = pd.read_excel(file_path, skiprows=2)
    
    # Extract data for each configuration using column letters
    milvus_llama = df.iloc[:, 2:8].copy()  # Columns C to H
    milvus_llama.columns = ['RMSE_Context_Rel', 'RMSE_Context_Util', 'AUCROC', 
                           'Retrieval_Time', 'Context_Relevance', 'Context_Utilization']
    
    weaviate_mistral = df.iloc[:, 9:16].copy()  # Columns J to P
    weaviate_mistral.columns = ['Retrieval_Time', 'Context_Rel', 'Util', 
                               'Adherence', 'RMSE_Context_Rel', 'RMSE_Context_Util', 'AUCROC']
    
    milvus_mistral = df.iloc[:, 17:24].copy()  # Columns R to X
    milvus_mistral.columns = ['Retrieval_Time', 'Context_Rel', 'Util', 
                             'Adherence', 'RMSE_Context_Rel', 'RMSE_Context_Util', 'AUCROC']
    
    # Replace 'na' with NaN and convert to float
    milvus_llama = milvus_llama.replace('na', np.nan).astype(float)
    weaviate_mistral = weaviate_mistral.replace('na', np.nan).astype(float)
    milvus_mistral = milvus_mistral.replace('na', np.nan).astype(float)
    
    return milvus_llama, weaviate_mistral, milvus_mistral

def create_performance_comparison(milvus_llama, weaviate_mistral, milvus_mistral):
    plt.style.use('default')  # Using default style instead of seaborn
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Retrieval Time Comparison
    data = {
        'Milvus + LLaMA': milvus_llama['Retrieval_Time'].dropna(),
        'Weaviate + Mistral': weaviate_mistral['Retrieval_Time'].dropna(),
        'Milvus + Mistral': milvus_mistral['Retrieval_Time'].dropna()
    }
    sns.boxplot(data=pd.DataFrame(data), ax=axes[0,0])
    axes[0,0].set_title('Retrieval Time Comparison')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # RMSE Context Relevance Comparison
    data = {
        'Milvus + LLaMA': milvus_llama['RMSE_Context_Rel'].dropna(),
        'Weaviate + Mistral': weaviate_mistral['RMSE_Context_Rel'].dropna(),
        'Milvus + Mistral': milvus_mistral['RMSE_Context_Rel'].dropna()
    }
    sns.boxplot(data=pd.DataFrame(data), ax=axes[0,1])
    axes[0,1].set_title('RMSE Context Relevance')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # RMSE Context Utilization Comparison
    data = {
        'Milvus + LLaMA': milvus_llama['RMSE_Context_Util'].dropna(),
        'Weaviate + Mistral': weaviate_mistral['RMSE_Context_Util'].dropna(),
        'Milvus + Mistral': milvus_mistral['RMSE_Context_Util'].dropna()
    }
    sns.boxplot(data=pd.DataFrame(data), ax=axes[1,0])
    axes[1,0].set_title('RMSE Context Utilization')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # AUROC Comparison
    data = {
        'Milvus + LLaMA': milvus_llama['AUCROC'].dropna(),
        'Weaviate + Mistral': weaviate_mistral['AUCROC'].dropna(),
        'Milvus + Mistral': milvus_mistral['AUCROC'].dropna()
    }
    sns.boxplot(data=pd.DataFrame(data), ax=axes[1,1])
    axes[1,1].set_title('AUROC Scores')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('report/visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmaps(milvus_llama, weaviate_mistral, milvus_mistral):
    plt.figure(figsize=(20, 6))
    
    # Create correlation heatmaps for each configuration
    plt.subplot(1, 3, 1)
    sns.heatmap(milvus_llama.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Milvus + LLaMA Correlations')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(weaviate_mistral.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Weaviate + Mistral Correlations')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(milvus_mistral.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Milvus + Mistral Correlations')
    
    plt.tight_layout()
    plt.savefig('report/visualizations/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_violin_plots(milvus_llama, weaviate_mistral, milvus_mistral):
    metrics = ['RMSE_Context_Rel', 'RMSE_Context_Util', 'AUCROC']
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        data = {
            'Milvus + LLaMA': milvus_llama[metric].dropna(),
            'Weaviate + Mistral': weaviate_mistral[metric].dropna(),
            'Milvus + Mistral': milvus_mistral[metric].dropna()
        }
        sns.violinplot(data=pd.DataFrame(data))
        plt.title(f'{metric} Distribution')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('report/visualizations/metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_statistics(milvus_llama, weaviate_mistral, milvus_mistral):
    print("\nSummary Statistics:")
    
    print("\nMilvus + LLaMA:")
    print(milvus_llama.describe().round(4))
    
    print("\nWeaviate + Mistral:")
    print(weaviate_mistral.describe().round(4))
    
    print("\nMilvus + Mistral:")
    print(milvus_mistral.describe().round(4))

def main():
    # Create visualizations directory
    import os
    os.makedirs("report/visualizations", exist_ok=True)
    
    # Load data
    milvus_llama, weaviate_mistral, milvus_mistral = load_and_preprocess_data("report/Scores for RAGBenchCapstone.xlsx")
    
    # Create visualizations
    create_performance_comparison(milvus_llama, weaviate_mistral, milvus_mistral)
    create_correlation_heatmaps(milvus_llama, weaviate_mistral, milvus_mistral)
    create_violin_plots(milvus_llama, weaviate_mistral, milvus_mistral)
    
    # Print statistics
    print_summary_statistics(milvus_llama, weaviate_mistral, milvus_mistral)

if __name__ == "__main__":
    main()