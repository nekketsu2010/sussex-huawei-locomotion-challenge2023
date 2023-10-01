import numpy as np
import os
import scipy
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import he_normal

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# get the norm of x, y, z
def norm(x):
    x_norm = np.linalg.norm(x[:, 1:], axis=1)
    x = np.concatenate([x[:, 0].reshape([-1, 1]), x_norm.reshape([-1, 1])], axis=1)
    return x

# Calculate spectrogram
def spectrogram(x, nfft=32, fs=50, axis=1):
    overlap = 4
    
    f, t, x = scipy.signal.spectrogram(x, fs=fs, nperseg=nfft,noverlap=overlap)
    
    # Standardize by specifying the axis (default axis is standardized on the frequency axis)
    x = (x - np.mean(x, axis=axis, keepdims=True)) / np.std(x, axis=axis, keepdims=True)
    x = np.round(x, 5)
    return f, t, x

def spliter(x):
    x_train, x_val = train_test_split(x, test_size=0.3, shuffle=False)
    return x_train, x_val

def spliter2(x, y):
    x_train, x_val = train_test_split(x, test_size=0.3, stratify=y)
    return x_train, x_val

def extract_Ns(data, change_indexes):
    sections = []

    sections.append(data[:change_indexes[0]+1])

    for i in range(1, len(change_indexes)):
        sections.append(data[change_indexes[i-1]+1:change_indexes[i]+1])

    sections.append(data[change_indexes[-1]+1:])

    # Extract every N points
    N = 500
    result = []
    for i in range(len(sections)):
        for j in range(0, sections[i].shape[0], N):
            if not sections[i][j:j+N].shape[0] == N:
                continue
            result.append(sections[i][j:j+N])
    result = np.array(result)
    
    return result

def BuildModel(input_shape):
    input1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 5), padding='valid', activation='relu', kernel_initializer=he_normal())(input1)
    x = layers.Conv2D(32, (3, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 5), padding='same', activation='relu')(x)
    x = layers.Conv2D(32, (3, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = models.Model(inputs=input1, outputs=x)
    return x


VAL_PATH = '../../data/npy/validate/'
TEST_PATH = '../../data/npy/test/'

save_folder = "../../output/checkpoint/" #save directory

def main():
    acc = np.load(VAL_PATH + 'Hand/GloAcc_ver2.npy')
    gyr = np.load(VAL_PATH + 'Hand/GloGyr_ver2.npy')
    mag = np.load(VAL_PATH + 'Hand/GloMag_ver2.npy')
    label = np.load(VAL_PATH + 'Hand/Label.npy')

    acc[:, 1:] = (acc[:, 1:] - np.mean(acc[:, 1:], axis=0, keepdims=True)) / np.std(acc[:, 1:], axis=0, keepdims=True)
    gyr[:, 1:] = (gyr[:, 1:] - np.mean(gyr[:, 1:], axis=0, keepdims=True)) / np.std(gyr[:, 1:], axis=0, keepdims=True)
    mag[:, 1:] = (mag[:, 1:] - np.mean(mag[:, 1:], axis=0, keepdims=True)) / np.std(mag[:, 1:], axis=0, keepdims=True)

    change_indexes = np.where(np.abs(np.diff(acc[:, 0])) > 1.5e+8)[0] + 1

    user2_acc = acc[:change_indexes[2]]
    user2_gyr = gyr[:change_indexes[2]]
    user2_mag = mag[:change_indexes[2]]
    user2_label = label[:change_indexes[2]]

    user3_acc = acc[change_indexes[2]:]
    user3_gyr = gyr[change_indexes[2]:]
    user3_mag = mag[change_indexes[2]:]
    user3_label = label[change_indexes[2]:]

    user2_acc[:, 1:] = np.round(user2_acc[:, 1:], 5)
    user2_gyr[:, 1:] = np.round(user2_gyr[:, 1:], 5)
    user2_mag[:, 1:] = np.round(user2_mag[:, 1:], 5)

    user3_acc[:, 1:] = np.round(user3_acc[:, 1:], 5)
    user3_gyr[:, 1:] = np.round(user3_gyr[:, 1:], 5)
    user3_mag[:, 1:] = np.round(user3_mag[:, 1:], 5)

    timestamp = user2_acc[::2, 0]
    user2_acc = scipy.signal.decimate(user2_acc, 2, axis=0)
    user2_acc[:, 0] = timestamp
    user2_gyr = scipy.signal.decimate(user2_gyr, 2, axis=0)
    user2_gyr[:, 0] = timestamp
    user2_mag = scipy.signal.decimate(user2_mag, 2, axis=0)
    user2_mag[:, 0] = timestamp
    user2_label = user2_label[::2]

    timestamp = user3_acc[::2, 0]
    user3_acc = scipy.signal.decimate(user3_acc, 2, axis=0)
    user3_acc[:, 0] = timestamp
    user3_gyr = scipy.signal.decimate(user3_gyr, 2, axis=0)
    user3_gyr[:, 0] = timestamp
    user3_mag = scipy.signal.decimate(user3_mag, 2, axis=0)
    user3_mag[:, 0] = timestamp
    user3_label = user3_label[::2]

    user2_acc = norm(user2_acc)
    user2_gyr = norm(user2_gyr)
    user2_mag = norm(user2_mag)

    user3_acc = norm(user3_acc)
    user3_gyr = norm(user3_gyr)
    user3_mag = norm(user3_mag)

    # Segment every 10 seconds
    # It is 500 points because it is downsampled to 50Hz.
    user2_change_indexes = np.where(np.abs(np.diff(user2_acc[:, 0])) > 20)[0]
    user3_change_indexes = np.where(np.abs(np.diff(user3_acc[:, 0])) > 20)[0]

    user2_acc = extract_Ns(user2_acc, user2_change_indexes)
    user2_gyr = extract_Ns(user2_gyr, user2_change_indexes)
    user2_mag = extract_Ns(user2_mag, user2_change_indexes)
    user2_label = extract_Ns(user2_label, user2_change_indexes)

    user3_acc = extract_Ns(user3_acc, user3_change_indexes)
    user3_gyr = extract_Ns(user3_gyr, user3_change_indexes)
    user3_mag = extract_Ns(user3_mag, user3_change_indexes)
    user3_label = extract_Ns(user3_label, user3_change_indexes)

    user2_acc_spectrogram = spectrogram(user2_acc[:, :, 1])[-1]
    user2_gyr_spectrogram = spectrogram(user2_gyr[:, :, 1])[-1]
    user2_mag_spectrogram = spectrogram(user2_mag[:, :, 1])[-1]

    user3_acc_spectrogram = spectrogram(user3_acc[:, :, 1])[-1]
    user3_gyr_spectrogram = spectrogram(user3_gyr[:, :, 1])[-1]
    user3_mag_spectrogram = spectrogram(user3_mag[:, :, 1])[-1]

    user2_user_label = np.zeros(len(user2_acc_spectrogram), dtype=np.int8)
    user2_user_label[:] = 0

    user3_user_label = np.zeros(len(user3_acc_spectrogram), dtype=np.int8)
    user3_user_label[:] = 1

    user2 = np.concatenate([
        np.expand_dims(user2_acc_spectrogram, [1, -1]),
        np.expand_dims(user2_gyr_spectrogram, [1, -1]),
        np.expand_dims(user2_mag_spectrogram, [1, -1]),
    ], axis=1)

    user3 = np.concatenate([
        np.expand_dims(user3_acc_spectrogram, [1, -1]),
        np.expand_dims(user3_gyr_spectrogram, [1, -1]),
        np.expand_dims(user3_mag_spectrogram, [1, -1]),
    ], axis=1)

    user2_train, user2_val = spliter2(user2, user2_label[:, 0, 1])
    user3_train, user3_val = spliter2(user3, user3_label[:, 0, 1])

    user2_train_label, user2_val_label = spliter2(user2_label, user2_label[:, 0, 1])
    user3_train_label, user3_val_label = spliter2(user3_label, user3_label[:, 0, 1])

    user2_user_train_label, user2_user_val_label = spliter(user2_user_label)
    user3_user_train_label, user3_user_val_label = spliter(user3_user_label)

    x_train = np.concatenate([user2_train, user3_train])
    x_val = np.concatenate([user2_val, user3_val])

    y_train = np.concatenate([user2_user_train_label, user3_user_train_label])
    y_val = np.concatenate([user2_user_val_label, user3_user_val_label])

    user_train_label = np.concatenate([user2_train_label, user3_train_label])
    user_val_label = np.concatenate([user2_val_label, user3_val_label])

    x1 = BuildModel(x_train[0, 0].shape)
    x2 = BuildModel(x_train[0, 1].shape)
    x3 = BuildModel(x_train[0, 2].shape)

    combined = layers.concatenate([x1.output, x2.output, x3.output])

    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dense(16, activation='relu')(z)
    z = layers.Dense(2, activation='softmax')(z)

    model = models.Model(inputs=[x1.input, x2.input, x3.input], outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder + "model_20230617.hdf5", 
                                            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    history = model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2]], y_train, epochs=256, batch_size=64, \
                        validation_data=([x_val[:, 0], x_val[:, 1], x_val[:, 2]], y_val), callbacks=[cp_cb, es_cb])

    model = tf.keras.models.load_model(save_folder + 'model_20230617.hdf5')

    acc = np.load(TEST_PATH + 'GloAcc_ver2.npy')
    gyr = np.load(TEST_PATH + 'GloGyr_ver2.npy')
    mag = np.load(TEST_PATH + 'GloMag_ver2.npy')

    acc[:, 1:] = (acc[:, 1:] - np.mean(acc[:, 1:], axis=0, keepdims=True)) / np.std(acc[:, 1:], axis=0, keepdims=True)
    gyr[:, 1:] = (gyr[:, 1:] - np.mean(gyr[:, 1:], axis=0, keepdims=True)) / np.std(gyr[:, 1:], axis=0, keepdims=True)
    mag[:, 1:] = (mag[:, 1:] - np.mean(mag[:, 1:], axis=0, keepdims=True)) / np.std(mag[:, 1:], axis=0, keepdims=True)

    acc[:, 1:] = np.round(acc[:, 1:], 5)
    gyr[:, 1:] = np.round(gyr[:, 1:], 5)
    mag[:, 1:] = np.round(mag[:, 1:], 5)

    timestamp = acc[::2, 0]
    acc = scipy.signal.decimate(acc, 2, axis=0)
    acc[:, 0] = timestamp
    gyr = scipy.signal.decimate(gyr, 2, axis=0)
    gyr[:, 0] = timestamp
    mag = scipy.signal.decimate(mag, 2, axis=0)
    mag[:, 0] = timestamp

    acc = norm(acc)
    gyr = norm(gyr)
    mag = norm(mag)

    change_indexes = np.where(np.abs(np.diff(acc[:, 0])) > 20)[0]

    acc = extract_Ns(acc, change_indexes)
    gyr = extract_Ns(gyr, change_indexes)
    mag = extract_Ns(mag, change_indexes)

    acc_spectrogram = spectrogram(acc[:, :, 1])[-1]
    gyr_spectrogram = spectrogram(gyr[:, :, 1])[-1]
    mag_spectrogram = spectrogram(mag[:, :, 1])[-1]

    x_test = np.concatenate([
        np.expand_dims(acc_spectrogram, [1, -1]),
        np.expand_dims(gyr_spectrogram, [1, -1]),
        np.expand_dims(mag_spectrogram, [1, -1]),
    ], axis=1)

    class_names = ['User2', 'User3']
    predict = model.predict([x_test[:, 0], x_test[:, 1], x_test[:, 2]])
    predict = np.argmax(predict, axis=1)

    plt.figure(figsize=(24, 8))

    for i in tqdm(range(len(acc))):
        plt.plot(list(range(len(acc)))[i:i+2], acc[i:i+2, 0, 0], color='r' if predict[i] == 0 else 'b')
        
    plt.show()

if __name__ == '__main__':
    main()
    