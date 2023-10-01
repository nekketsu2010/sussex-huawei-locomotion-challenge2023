TRAIN_TXT_DIR="/raw/SHL-2023-Train/train/"
VAL_TXT_DIR="/raw/SHL-2023-Validate/validate/"
TEST_TXT_DIR="/raw/SHL-2023-Test/test/"
RAW_FILENAMES = ['Acc', 'Gyr', 'Mag', 'Location', 'GPS']
SENSOR_IN = ['Acc.npy','Gyr.npy','Mag.npy']
TYPE = ['train','validate','test']
POS = ['Bag','Hand','Hips','Torso']
OUTPUT_FILENAME ='Magnitude.npy'
LABEL_FILENAME = 'Label.npy'
LABEL_FILENAME_TEST = 'Label_idx.npy'


RAWNPY_DIR = 'npy'
RAWNPY_DIR_500 = 'every_5s'
MAGNITUDE_DIR = 'every_5s_magnitude'
F_SIMPLE_LABEL_DIR='features/simple_label'
INPUT_ROOT = '../'

CHANGE_INDEXES_FILE = 'ChangeIndexes.npy'
CHANGE_INDEXES_DIR = 'change_indexes'
OUTPUT_ROOT = '../output/'
OUTPUT_DIR  = '../output/data/every_5s'