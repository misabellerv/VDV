import argparse
from workflow.run_workflow import run_workflow
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)

    args = parser.parse_args()

    if args.config is not None:
        config = int(args.config)
    else:
        fold = None

    run_workflow(config_path="configs/configs.json")