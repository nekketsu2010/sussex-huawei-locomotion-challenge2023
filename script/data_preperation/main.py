import yaml
from const_prep import TYPE,POS,INPUT_DIR_500,MAGNITUDE_DIR,F_SIMPLE_LABEL_DIR
from extract_windows import extract_window
from data_preparation import extract_features,extract_magnitude

def main(config):
    for type in TYPE:
        extract_window(config,type,positions=POS,n=500)
        extract_magnitude(config,type,INPUT_DIR_500,MAGNITUDE_DIR)
        extract_features(config,type,MAGNITUDE_DIR,F_SIMPLE_LABEL_DIR)

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)