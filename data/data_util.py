import os
import yaml

def parse_from_yaml(file_pth):
    with open(file_pth, 'r') as file:
        config = yaml.safe_load(file)
        # print(config)
        # print("YAML file has been parsed!!!!!")
        return config