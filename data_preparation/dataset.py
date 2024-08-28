import pandas as pd
import time

class DataPreparation():
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        
        print('Loading Data...')
        
        start_time = time.time()
        
        X_train_path = self.config.DataPaths.TrainImages
        y_train_path = self.config.DataPaths.TrainLabels
        X_test_path = self.config.DataPaths.TestImages
        y_test_path = self.config.DataPaths.TestLabels
        
        X_train = pd.read_csv(X_train_path, header=None)
        X_test = pd.read_csv(X_test_path, header=None)
        y_train = pd.read_csv(y_train_path)
        y_train = y_train['Volcano?']
        y_test = pd.read_csv(y_test_path)
        y_test = y_test['Volcano?']
        
        print(f'Data Loading Finished in {time.time() - start_time:.2f} Seconds!')
        
        return X_train, X_test, y_train, y_test