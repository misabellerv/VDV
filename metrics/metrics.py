from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def get_metrics(target, predictions):
    
    print('Generating Metrics...')

    accuracy = accuracy_score(target, predictions)
    precision = precision_score(target, predictions, average='weighted')
    recall = recall_score(target, predictions, average='weighted')
    f1 = f1_score(target, predictions, average='weighted')
        
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    
    print('Metrics Sucessefully Generated!')
    return metrics_df