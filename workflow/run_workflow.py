import warnings
import time
from config import get_config
from data_preparation.dataset import DataPreparation
from data_preparation.split_data import data_split
from preprocessing.pipeline import preprocess_pipeline
from model.models import ModelSelection
from model.grid_seach import GSCVTrainer
from model.param_grid import extract_param_grid
from model.train import Trainer
from model.get_hyperparam import get_best_hyperparam
warnings.filterwarnings('ignore')

def run_workflow(config_path):
    
    start_time = time.time()
    
    # Get Config File
    config = get_config(config_path)

    # Load Data
    data_prep = DataPreparation(config)
    X_train, X_test, y_train, y_test = data_prep.load_data()
    
    # Data augmentation
    
    

    # Pre Process Data
    X_proc_train = preprocess_pipeline(X_train, config)

    # Data Split (training/validation)
    X_train_, y_train_, X_val, y_val = data_split(config=config, dataframe=X_proc_train, target=y_train)

    # Get Models
    model = ModelSelection(config)

    svm = model.svm()
    dt = model.dt()
    xgb = model.xgb()
    knn = model.knn()

    # Extract Grid Parameter
    param_svm, param_dt, param_xgb, param_knn = extract_param_grid(config)

    # GridSearchCV Training
    trainer_svm = GSCVTrainer(model=svm, param_grid=param_svm, config=config)
    trainer_dt = GSCVTrainer(model=dt, param_grid=param_dt, config=config)
    trainer_xgb = GSCVTrainer(model=xgb, param_grid=param_xgb, config=config)
    trainer_knn = GSCVTrainer(model=knn, param_grid=param_knn, config=config)

    trainer_svm.fit(X_train_, y_train_)
    trainer_dt.fit(X_train_, y_train_)
    trainer_xgb.fit(X_train_, y_train_)
    trainer_knn.fit(X_train_, y_train_)

    # Predict on Validation Set
    trainer_svm.predict(X_val,y_val)
    trainer_dt.predict(X_val,y_val)
    trainer_xgb.predict(X_val,y_val)
    trainer_knn.predict(X_val, y_val)

    # train+val Training using Best Hyperparameters
    svm_hyperparam = get_best_hyperparam('SVC')
    model_train_val_svm = Trainer(model=svm, params=svm_hyperparam, config=config)
    model_train_val_svm.fit(X_train, y_train)
    
    dt_hyperparam = get_best_hyperparam('DecisionTreeClassifier')
    model_train_val_dt = Trainer(model=dt, params=dt_hyperparam, config=config)
    model_train_val_dt.fit(X_train, y_train)
    
    xgb_hyperparam = get_best_hyperparam('XGBClassifier')
    model_train_val_xgb = Trainer(model=xgb, params=xgb_hyperparam, config=config)
    model_train_val_xgb.fit(X_train, y_train)
    
    knn_hyperparam = get_best_hyperparam('KNeighborsClassifier')
    model_train_val_knn = Trainer(model=knn, params=knn_hyperparam, config=config)
    model_train_val_knn.fit(X_train, y_train)

    # Predict on Test Set
    model_train_val_svm.predict(X_test, y_test)
    model_train_val_dt.predict(X_test, y_test)
    model_train_val_xgb.predict(X_test, y_test)
    model_train_val_knn.predict(X_test, y_test)
    
    print(f'Full Workflow Completed in {time.time() - start_time:.2f} Seconds.')