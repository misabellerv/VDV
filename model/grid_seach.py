import json
from sklearn.model_selection import GridSearchCV
import gc
import os
import joblib
import time
from metrics.metrics import get_metrics
from visualization.plots import plot_confusion_matrix

class GSCVTrainer:
    def __init__(self, model, param_grid, config):
        self.model = model
        self.param_grid = param_grid
        self.cv = config.Training.GridSearchCV.CV
        self.scoring = config.Training.GridSearchCV.Scoring
        self.parallel = config.Joblib.Parallelize
        self.n_jobs = config.Joblib.NumJobs
        self.verbose = config.Training.GridSearchCV.Verbose
        self.grid_search = None
        self.best_model_path = None  

    def fit(self, X_train, y_train):
        model_name = self.model.__class__.__name__
        print(f"Starting {model_name} Training...")
        start_time = time.time()
        
        if self.parallel: 
            self.grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            self.grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                verbose=self.verbose
            )
        
        self.grid_search.fit(X_train, y_train)
        print(f"Model Training Finished in {time.time() - start_time:.2f} Seconds!")

        results_path = 'results/model'
        os.makedirs(results_path, exist_ok=True)
        
        best_params_path = results_path + f'/{model_name}_best_params.json'
        with open(best_params_path, 'w') as json_file:
            json.dump(self.grid_search.best_params_, json_file)

        self.best_model_path = f'results/model/{model_name}_best_model.pkl'
        joblib.dump(self.grid_search.best_estimator_, self.best_model_path)
        
        del X_train, y_train
        gc.collect()
        
        return self.grid_search.best_estimator_

    def predict(self, X_test, y_test):
        model_name = self.model.__class__.__name__

        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}...")
            best_model = joblib.load(self.best_model_path) 
        else:
            if self.grid_search is None:
                raise ValueError("The model has not been trained yet. Train the model first or provide a trained model.")
            best_model = self.grid_search.best_estimator_  
        
        print("Starting Predictions...")
        self.predictions = best_model.predict(X_test) 
        metrics_df = get_metrics(y_test, self.predictions)

        metrics = 'results/metrics'
        os.makedirs(metrics, exist_ok=True)
        metrics_df.to_csv(metrics + f'/{model_name}_metrics.csv')

        c_matrix = 'results/plots'
        os.makedirs(c_matrix, exist_ok=True)

        if hasattr(best_model, 'classes_'):
            labels = best_model.classes_
        else:
            labels = sorted(set(y_test))
            print(f"Model does not have 'classes_' attribute. Using labels from y_test: {labels}")
        
        plot_confusion_matrix(y_test, self.predictions, labels, os.path.join(c_matrix, f'{model_name}_val_confusion_matrix.png'))

        print("Prediction Results Finished!")
        
        del X_test, y_test
        gc.collect()
        
        return self.predictions
    
