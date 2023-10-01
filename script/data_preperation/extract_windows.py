import os 
import numpy as np
# import cupy as cp
from tqdm import tqdm
import pathlib
from const_prep import TYPE,CHANGE_INDEXES_DIR,CHANGE_INDEXES_FILE,LABEL_FILENAME,POS,SENSOR_IN,DIRNAME,LABEL_FILENAME_TEST,RAWNPY_DIR,RAWNPY_DIR_500

# def file_exist(path):
# path='path/to/file/file.ext'のファイルが存在するかどうかをTrue,Falseで返す
# 存在しますのログ付き
# あんまりいらないかも
def file_exist(path):
    if os.path.isfile(path):
        print(path+"は存在します")
        return True
    else:
        return False
    
# def savenpy(path, obj):
# np.arrayをpathに保存
# pathまでのディレクトリも作る
def savenpy(path,ary):
    #ディレクトリを作って保存
    os.makedirs(os.path.dirname(path),exist_ok=True)
    np.save(path, ary)  

# ['train','validate','test'] => [0,1,2] に変換
def type2idx(type):
    for i, elem in enumerate(TYPE):
        if elem == type:
            return i
    return None

# def extract_change_index(type):
# type は ['train','validate','test'] のstringで渡す
# change_indexes(2秒以上離れているタイムスタンプのindex)を抽出
def extract_change_index(config,type):
    idx = type2idx(type)
    output_path = os.path.join(config['data_dir'],CHANGE_INDEXES_DIR,TYPE[idx],CHANGE_INDEXES_FILE)
    if idx == 2:
        label = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],LABEL_FILENAME_TEST))
        change_indexes = np.where(np.abs(np.diff(label) > 10))[0]
    elif idx == 0:
        TRAINING_PATH = '../data/npy/train/'
        label = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],POS[0],LABEL_FILENAME))
        acc = np.load(TRAINING_PATH +'Hand/Acc.npy')
        mag = np.load(TRAINING_PATH +'Hand/Mag.npy')
        gyr = np.load(TRAINING_PATH +'Hand/Gyr.npy')
        change_indexes = np.where((np.abs(np.diff(label[:, 0])) > 10) |
                                  (np.isnan(acc[:-1,1]))| (np.isnan(mag[:-1,1]))| (np.isnan(gyr[:-1,1])))[0]
    else:
        label = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],POS[0],LABEL_FILENAME))
        change_indexes = np.where((np.abs(np.diff(label[:, 0])) > 10))[0]
    savenpy(output_path,change_indexes)

def extract_window(config,type,positions=['Hand'],n=500):
    idx = type2idx(type)
    # testは別関数へ
    if idx == 2:
        extract_window_test(config,n)
        return
    if not file_exist(os.path.join(config['data_dir'],CHANGE_INDEXES_DIR,TYPE[idx],CHANGE_INDEXES_FILE)):
        extract_change_index(TYPE[idx])
    change_indexes = np.load(os.path.join(config['data_dir'],CHANGE_INDEXES_DIR,TYPE[idx],CHANGE_INDEXES_FILE))
    # ラベル
    label = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],POS[0],LABEL_FILENAME))
    result = extract_Ns(label,change_indexes,n)
    print(result.shape)
    # ラベルがN個の中で変化しているサンプルを除外
    u,i = np.unique(result[:,:,1],return_index=True,axis=0)
    mask = ~np.all(u == u[:,0][:,np.newaxis],axis=1)
    # mask
    exclude_samples = i[mask]
    #　保存
    if not file_exist(os.path.join(config['data_dir'],RAWNPY_DIR_500,TYPE[idx],LABEL_FILENAME)):
        result = np.delete(result,exclude_samples,0)
        output_path = os.path.join(config['data_dir'],RAWNPY_DIR_500,TYPE[idx],LABEL_FILENAME)
        savenpy(output_path,result)
    # 保持位置ごとに
    for pos in positions:
        #　保持位置のフォルダ内のファイルごとに
        for f in pathlib.Path(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],pos)).glob('*.npy'):
            filename = f.name
            if file_exist(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],pos,filename)):
                continue
            print(filename,end=" ")
            # 形式が違うためスキップ
            if filename=='Location.npy' or filename=='GPS.npy':
                print("is skipped")
                continue
            # ロード
            result = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],pos,filename))
            #　分割
            result = extract_Ns(result,change_indexes,n)
            print(result.shape,change_indexes.shape,end=" ")
            #　ラベルが複数種類のサンプルを除外
            result = np.delete(result,exclude_samples,0)
            print("after deletion:",result.shape)
            #　保存
            output_path = os.path.join(config['data_dir'],RAWNPY_DIR_500,TYPE[idx],pos,filename)
            savenpy(output_path,result)

def extract_window_test(n,config):
    idx = 2
    if not file_exist(os.path.join(config['data_dir'],CHANGE_INDEXES_DIR,TYPE[idx],CHANGE_INDEXES_FILE)):
        extract_change_index(TYPE[idx])
    change_indexes = np.load(os.path.join(config['data_dir'],CHANGE_INDEXES_DIR,TYPE[idx],CHANGE_INDEXES_FILE))
    for f in pathlib.Path(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx])).glob('*.npy'):
        filename = f.name
        print(filename,end=" ")
        if filename=='Location.npy' or filename=='GPS.npy':
            print("is skipped")
            continue
        if file_exist(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],filename)):
            continue
        data = np.load(os.path.join(config['data_dir'],RAWNPY_DIR,TYPE[idx],filename))
        result =extract_Ns(data,change_indexes,n,is_test=True)
        print(result.shape,change_indexes.shape)
        savenpy(os.path.join(config['data_dir'],RAWNPY_DIR_500,TYPE[idx],filename),result)
                
def extract_Ns(data, change_indexes,n,is_test=False):
    sections = []

    sections.append(data[:change_indexes[0]+1])

    for i in range(1, len(change_indexes)):
        sections.append(data[change_indexes[i-1]+1:change_indexes[i]+1])

    sections.append(data[change_indexes[-1]+1:])

    # Nポイントずつ取り出す
    result = []
    for i in range(len(sections)):
        for j in range(0, sections[i].shape[0], n):
            if not sections[i][j:j+n].shape[0] == n:
                if not is_test:
                    continue
                if len(sections[i][j:j+n].shape) == 2:
                    pad_arg = ((0, n - sections[i][j:j+n].shape[0]), (0, 0))
                elif len(sections[i][j:j+n].shape) == 1:
                    pad_arg = ((0, n - sections[i][j:j+n].shape[0]))
                x = np.pad(sections[i][j:j+n], pad_arg, constant_values=(np.nan))
                result.append(x)
                continue
            # if sections[i].shape[1] == 2:
            #     if len(np.unique(sections[i][j:j+N, 1])) > 1:
            #         continue
            result.append(sections[i][j:j+n])
    result = np.array(result)
    
    return result

# extract_change_index('train')
# extract_change_index('validate')
# extract_change_index('test')
# extract_500('train')
# extract_500('validate')
# extract_500('test')
# extract_window('train',500,positions=POS)
# extract_window('validate',500,positions=POS)
# extract_window('test',500,positions=POS)
# extract_window('train',500)
# extract_window('validate',500,positions=POS)
# extract_window('test',500)