import time
from sklearn.model_selection import train_test_split


def data_split(config, dataframe, target):
    
    print('Starting Data Splitting...')
    
    start_time = time.time()
    
    test_size = config.Training.GridSearchCV.TestSize
    seed = config.Training.Seed
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target,
                                                      test_size=test_size,
                                                      random_state=seed)
    print(f'Finished Data Splitting in {time.time() - start_time:.2f} seconds!')
    return X_train, y_train, X_test, y_test