def extract_param_grid(config):
    """
    Extract parameter grids for SVM, Decision Trees, XGBoost and KNN from the config file.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        tuple: A tuple containing parameter grids for SVM, Decision Trees, XGBoost and KNN
    """
    param_grid_svm = {
        'C': config.Models.SVM.C,
        'kernel': config.Models.SVM.Kernel
    }

    param_grid_dt = {
        'max_depth': config.Models.DecisionTrees.MaxDepth,
        'min_samples_split': config.Models.DecisionTrees.MinSamplesSplit,
        'criterion': config.Models.DecisionTrees.Criterion
    }

    param_grid_xgb = {
        'n_estimators': config.Models.XGBoost.Nestimators,
        'learning_rate': config.Models.XGBoost.LearningRate,
        'use_label_encoder': config.Models.XGBoost.UseLabelEncoder,
    }
    
    param_grid_knn = {
        'n_neighbors': config.Models.KNN.Neighbors,
        'metric': config.Models.KNN.Metric
    }

    return param_grid_svm, param_grid_dt, param_grid_xgb, param_grid_knn