import yaml
from txt2npy import exec_txt2npy
from changeIndexes import exec_extract_change_indexes
from convertAcc import exec_convertAcc
from convertWorld_by_Android import exec_convert_world
from convert_location import exec_convert_location

def main(config):
    exec_txt2npy(config)
    exec_extract_change_indexes(config)
    exec_convertAcc(config)
    exec_convert_world()
    exec_convert_location()

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)