import os
import json
from dotwiz import DotWiz

def get_config(config_path=''):

    # This class loads and returns configs from config_files 
    assert isinstance(config_path, str) == True, "Config:: config_path should be a string, " + str(type(config_path)) + " found"
    assert os.path.isfile(config_path) == True, f"Config:: File {config_path} not found in config_path"
    
    # Load properties file
    with open(config_path, 'r') as config_file:
        contents = json.loads(config_file.read())  

        # Return properties accessible by using dot notation
        properties = DotWiz(contents)
        properties.HomePath = os.path.abspath(os.path.dirname(__file__))
        properties.ConfigPath = os.path.abspath(config_path)

    return properties