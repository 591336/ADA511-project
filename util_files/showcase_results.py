import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hello():
    print("hello")

def show_results(models_results, model_names):
    """
    Showcase the results of multiple models, including consolidated printouts and plots.
    
    Parameters:
    - models_results: List of dictionaries containing model evaluation metrics (e.g., from evaluate_model()).
    - model_names: List of names corresponding to the models.
    """
    # Consolidate key metrics into a DataFrame for display
    metrics_df = pd.DataFrame(models_results)
    metrics_df['Model'] = model_names
    metrics_df = metrics_df.set_index('Model')

    # Display summary table
    print("\n=== Consolidated Performance Table ===")
    print(metrics_df[['total_utility', 'average_utility', 'sensitivity', 'specificity']])
    print("\n* Note: Not all metrics may be applicable to every model (e.g., 'total_utility' may be N/A for Standard DT)")

    # Bar chart for key metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    sns.barplot(data=metrics_df, x=metrics_df.index, y='total_utility', ax=axes[0], palette='viridis')
    axes[0].set_title('Total Utility Comparison')
    axes[0].set_ylabel('Total Utility')
    axes[0].set_xlabel('Model')

    sns.barplot(data=metrics_df, x=metrics_df.index, y='sensitivity', ax=axes[1], palette='viridis')
    axes[1].set_title('Sensitivity (Recall) Comparison')
    axes[1].set_ylabel('Sensitivity')
    axes[1].set_xlabel('Model')

    sns.barplot(data=metrics_df, x=metrics_df.index, y='specificity', ax=axes[2], palette='viridis')
    axes[2].set_title('Specificity Comparison')
    axes[2].set_ylabel('Specificity')
    axes[2].set_xlabel('Model')

    plt.tight_layout()
    plt.show()

    # Confusion matrix plots
    for i, result in enumerate(models_results):
        cm = result['confusion_matrix']
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title(f'{model_names[i]} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
