from config import get_config

def get_best_hyperparam(model_name):
    hyperparam = f'results/model/{model_name}_best_params.json'
    return get_config(hyperparam)
    