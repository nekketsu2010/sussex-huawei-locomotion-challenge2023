import numpy as np
import pandas as pd

# 移動状態ごとに7:3に分ける
def train_test_split_shl(x, y, train_size=0.67):
    x_train = np.empty((0,) + x.shape[1:])
    x_test = np.empty((0,) + x.shape[1:])
    y_train = np.empty((0,) + y.shape[1:])
    y_test = np.empty((0,) + y.shape[1:])
    
    for i in range(8):
        indexes = np.where(y[:, 1] == i+1)[0]
        train_index = int(len(indexes)*train_size)
        x_train = np.concatenate([x_train, x[indexes[:train_index]]], axis=0)
        x_test = np.concatenate([x_test, x[indexes[train_index:]]], axis=0)
        y_train = np.concatenate([y_train, y[indexes[:train_index]]], axis=0)
        y_test = np.concatenate([y_test, y[indexes[train_index:]]], axis=0)
        
    return x_train, x_test, y_train, y_test

OUTPUT_DIR = '../output/'
XGB_DIR = '../output/'
LSTM_DIR = '../output/0624_finetune8/'

def exec_post_processing():
    val_label = np.load(OUTPUT_DIR + 'validate/Label_every_5s.npy')[:, 0]

    val_lstm = np.load(LSTM_DIR + 'pred_val.npy')
    val_xgb = np.load(XGB_DIR + 'val_predict_xgboost_0626_001.npy')

    test_lstm = np.load(LSTM_DIR + 'pred_test3.npy')
    test_xgb = np.load(XGB_DIR + 'test_predict_xgboost_0626_001.npy')

    thresh = 1.5e+8
    change_indexes = np.where(np.abs(np.diff(val_xgb[:, 0])) > thresh)[0] + 1

    user2_val_label = val_label[:change_indexes[2]]
    user3_val_label = val_label[change_indexes[2]:]

    user2_val_lstm = val_lstm[:change_indexes[2]]
    user2_val_xgb = val_xgb[:change_indexes[2]]

    user3_val_lstm = val_lstm[change_indexes[2]:]
    user3_val_xgb = val_xgb[change_indexes[2]:]

    # train_test_splitで移動状態を考慮して7:3に分ける

    user2_val_lstm_train, user2_val_lstm_val, user2_val_label_train, user2_val_label_val \
        = train_test_split_shl(user2_val_lstm, user2_val_label)

    user2_val_xgb_train, user2_val_xgb_val, user2_val_label_train, user2_val_label_val \
        = train_test_split_shl(user2_val_xgb, user2_val_label)

    user3_val_lstm_train, user3_val_lstm_val, user3_val_label_train, user3_val_label_val \
        = train_test_split_shl(user3_val_lstm, user3_val_label)

    user3_val_xgb_train, user3_val_xgb_val, user3_val_label_train, user3_val_label_val \
        = train_test_split_shl(user3_val_xgb, user3_val_label)

    # 学習データ（EpochTimeは特徴にしないこと！含まれている場合は除外する）
    user2_train = np.concatenate([user2_val_lstm_train[:, 1:], user2_val_xgb_train[:, 1:]], axis=1)
    user3_train = np.concatenate([user3_val_lstm_train[:, 1:], user3_val_xgb_train[:, 1:]], axis=1)
    X_train = np.concatenate([user2_train, user3_train], axis=0)
    y_train = np.concatenate([user2_val_label_train, user3_val_label_train], axis=0)

    # 検証データ
    user2_val = np.concatenate([user2_val_lstm_val[:, 1:], user2_val_xgb_val[:, 1:]], axis=1)
    user3_val = np.concatenate([user3_val_lstm_val[:, 1:], user3_val_xgb_val[:, 1:]], axis=1)
    X_val = np.concatenate([user2_val, user3_val], axis=0)
    y_val = np.concatenate([user2_val_label_val, user3_val_label_val], axis=0)

    val_predict = np.load(OUTPUT_DIR + 'pred_val_ensemble_learning.npy')

    # ラベル順から時系列にソート（ユーザーごとに行うことに注意）
    val_predict[:len(user2_val_lstm_val)] = val_predict[:len(user2_val_lstm_val)][np.argsort(val_predict[:len(user2_val_lstm_val), 0])]
    val_predict[len(user2_val_lstm_val):] = val_predict[len(user2_val_lstm_val):][np.argsort(val_predict[len(user2_val_lstm_val):, 0])]

    # セクションごとに分ける
    change_indexes = np.where(np.abs(np.diff(val_predict[:, 0])) > 5000)[0] + 1
    change_indexes = np.insert(change_indexes, 0, 0)
    change_indexes = np.append(change_indexes, len(val_predict))

    from scipy.stats import mode

    POST_TIME = 96 # 5秒*96サンプル=480秒（8分）
    post_predict = val_predict.copy()
    for i in range(1, len(change_indexes)):
        start = change_indexes[i-1]
        end = change_indexes[i]
        start_tmp = change_indexes[i-1]
        end_tmp = start_tmp + POST_TIME
        while True:
            if end_tmp > end:
                end_tmp = end
            
            post_predict[start_tmp:end_tmp, 1] = mode(val_predict[start_tmp:end_tmp, 1])[0][0]
            
            if end_tmp == end:
                break
            else:
                start_tmp += POST_TIME
                end_tmp += POST_TIME
        
    # ラベル順から時系列にソート（ユーザーごとに行うことに注意）
    y_val[:len(user2_val_lstm_val)] = y_val[:len(user2_val_lstm_val)][np.argsort(y_val[:len(user2_val_lstm_val), 0])]
    y_val[len(user2_val_lstm_val):] = y_val[len(user2_val_lstm_val):][np.argsort(y_val[len(user2_val_lstm_val):, 0])]

    # Post-processing Test
    predict = np.load(OUTPUT_DIR + 'pred_test_ensemble_learning.npy')
    # セクションごとに分ける
    change_indexes = np.where(np.diff(np.abs(predict[:, 0])) > 5000)[0] + 1
    change_indexes = np.insert(change_indexes, 0, 0)
    change_indexes = np.append(change_indexes, len(predict))

    from scipy.stats import mode

    POST_TIME = 96 # 5秒*96サンプル=480秒（8分）
    post_predict = predict.copy()
    for i in range(1, len(change_indexes)):
        start = change_indexes[i-1]
        end = change_indexes[i]
        start_tmp = change_indexes[i-1]
        end_tmp = start_tmp + POST_TIME
        while True:
            if end_tmp > end:
                end_tmp = end
            
            post_predict[start_tmp:end_tmp, 1] = mode(predict[start_tmp:end_tmp, 1])[0][0]
            
            if end_tmp == end:
                break
            else:
                start_tmp += POST_TIME
                end_tmp += POST_TIME
        
    # pandasにする
    pd_predict = pd.DataFrame(post_predict)
    pd_predict = pd_predict.astype('int64')

    # フォーマットをなおす。
    test_submission = pd.read_csv('../data/raw/SHL-2023-example-submission/teamName_predictions.txt', sep='\t', header=None)
    test_submission = test_submission.drop(1, axis=1)

    test_submission = pd.merge_asof(test_submission, pd_predict, direction="nearest", on=0)
    test_submission.to_csv('../output/TDU_BSA_predictions.txt', sep='\t', header=None, index=None)

if __name__ == '__main__':
    exec_post_processing()
