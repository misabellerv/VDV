import pickle
import gc
from sklearn.metrics import classification_report
import pandas as pd
import os

class Trainer:
    def __init__(self, model, params, config):
        self.model = model
        self.params = params
        self.config = config

    def fit(self, X_train, y_train):
        model_name = self.model.__class__.__name__
        
        print(f"Starting {model_name} (train+val) Training...")

        self.model.set_params(**self.params)
        
        self.model.fit(X_train, y_train, verbose=True)
        print(f"{model_name} Training Finished!")
        

        results_path = 'results/model'
        os.makedirs(results_path, exist_ok=True)
        
        model_path = results_path + f'/{model_name}_train+val.pkl'
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        
        del X_train, y_train
        gc.collect()
        
        return self

    def predict(self, X_test, y_test):
        model_name = self.model.__class__.__name__
        if self.model is None:
            raise ValueError("The model has not been trained yet. Call `fit` method before `predict`.")
        
        print(f"Starting {model_name} Test Predictions...")
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        results = 'results/metrics'
        os.makedirs(results,exist_ok=True)
        
        df_report.to_csv(os.path.join(results, f'{model_name}_train+val_metrics.csv'))
            
        print(f"{model_name} Prediction Test Results Finished!")
        
        del X_test, y_test
        gc.collect()
        
        return predictions