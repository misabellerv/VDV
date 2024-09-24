import warnings
import time
from augmentation.apply_aug import augment_df
from config import get_config
from data_preparation.dataset import DataPreparation
from preprocessing.pipeline import preprocess_pipeline
from model.models import ModelSelection
from model.grid_seach import GSCVTrainer
from model.param_grid import extract_param_grid
warnings.filterwarnings('ignore')

def run_workflow(config_path):
    
    start_time = time.time()
    
    # Get Config File
    config = get_config(config_path)

    # Load Data
    data_prep = DataPreparation(config)
    X_train, X_test, y_train, y_test = data_prep.load_data()
    
   # Data augmentation
    X_train_aug, y_train_aug = augment_df(X_train, y_train)
    
    # Pre Process Data
    X_proc_train = preprocess_pipeline(X_train_aug, config)
    
    X_proc_test = preprocess_pipeline(X_test, config)

    # Get Models
    model = ModelSelection(config)

    svm = model.svm()
    knn = model.knn()
    xgb = model.xgb()
    dt = model.dt()

    # Extract Grid Parameter
    param_svm, param_dt, param_xgb, param_knn = extract_param_grid(config)

    # GridSearchCV Training
    trainer_svm = GSCVTrainer(model=svm, param_grid=param_svm, config=config)
    trainer_dt = GSCVTrainer(model=dt, param_grid=param_dt, config=config)
    trainer_xgb = GSCVTrainer(model=xgb, param_grid=param_xgb, config=config)
    trainer_knn = GSCVTrainer(model=knn, param_grid=param_knn, config=config)

    trainer_svm.fit(X_proc_train, y_train_aug)
    trainer_knn.fit(X_proc_train, y_train_aug)
    trainer_xgb.fit(X_proc_train, y_train_aug)
    trainer_dt.fit(X_proc_train, y_train_aug)
    
    # Predict on Test Set
    trainer_svm.predict(X_proc_test, y_test)
    trainer_knn.predict(X_proc_test, y_test)
    trainer_xgb.predict(X_proc_test, y_test)
    trainer_dt.predict(X_proc_test, y_test)
    
    print(f'Full Workflow Completed in {time.time() - start_time:.2f} Seconds.')
