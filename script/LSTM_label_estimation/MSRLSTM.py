import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import  Dense, Dropout, Layer,LSTM,Input, Concatenate,Conv1D,MaxPooling1D,Input,Multiply
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow_addons.optimizers import AdamW
import numpy as np
import math
import glob
import os

from utils import config

class ResidualConv(Layer):
    def __init__(self,batch_size, **kwargs):
        super(ResidualConv, self).__init__(**kwargs)
        self.conv1 = Conv1D(filters=64,kernel_size=3,strides=1,padding='same',input_shape=(batch_size,500))
        self.conv2 = Conv1D(filters=128,kernel_size=3,strides=1,padding='same')
        self.convsc = Conv1D(filters=128,kernel_size=4,strides=4,padding='same',input_shape=(batch_size,500))

        self.max = MaxPooling1D(pool_size=2,strides=2,padding='same')

    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.max(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.max(x)
        x = tf.nn.relu(x)
        sc = self.convsc(inputs)
        x += sc
        x = tf.nn.relu(x)
        x = self.max(x)
        return x
    
class MSRLSTM(Model):
    def __init__(self,
                batch_size=16,
                data_dim=9,
                data_sensor_num=3,
                lstm_dim=128,
                mlp_dim=[256,256,512,1024],
                dropout=0.2,
                num_classes=8,
                loc_dim=10
                ):
        super(MSRLSTM, self).__init__()
        self.dim = data_dim
        self.sensors = data_sensor_num
        self.batch_size = batch_size
        self.loc_dim = loc_dim
        self.res = [ResidualConv(batch_size=batch_size) for _ in range(self.dim)]
        self.concat1 = [Concatenate(axis=-1) for _ in range(self.dim)]
        self.concat2 = Concatenate(axis=-1)
        self.concat3 = Concatenate(axis=-1)
        self.conv = [
            Conv1D(filters=32,kernel_size=3,strides=1,padding='same')
            for _ in range(self.sensors)
        ]
        self.lstm = LSTM(lstm_dim, return_sequences=False)
        # self.lstm_peak = LSTM(lstm_dim, return_sequences=False)
        self.fc = Dense(128+loc_dim,activation='relu')
        self.fc2 = Dense(128+loc_dim,activation='softmax')
        self.mult = Multiply()
        self.mlp = [
            Dense(dim,activation='relu')
            for dim in mlp_dim
        ]
        self.final = Dense(num_classes,activation='softmax')
        self.dropout = [Dropout(dropout) for _ in range(len(mlp_dim))]

    def call(self, inputs, training):
        x = inputs[0]
        res_out = [[]for _ in range(self.dim//self.sensors)]
        for i in range(self.dim):
            res_out[i//self.sensors].append(self.res[i](x))
        conv_out = []
        for i in range(self.sensors):
            x = self.concat1[i](res_out[i])
            x = self.conv[i](x)
            conv_out.append(x)
        x = self.concat2(conv_out)
        x = self.lstm(x)

        lstm_out = self.concat3([x,inputs[1]])
        x = self.fc(lstm_out)

        x = self.fc2(x)

        x = self.mult([x,lstm_out])

        for i in range(len(self.mlp)):
            x = self.mlp[i](x)
            x = self.dropout[i](x,training=training)
        x = self.final(x)
        return x 
    
"""
using https://github.com/markub3327/HAR-Transformer/blob/b58d97c4d0fd7129770ce14596c752fd2b71a331/Training.ipynb#L350-L376
Copyright (c) 2022 Bc. Martin Kubovčík
License: https://github.com/markub3327/HAR-Transformer/blob/main/LICENSE 
"""
def cosine_schedule(base_lr, total_steps, warmup_steps):
 
    def step_fn(epoch):
        lr = base_lr
        epoch += 1

        progress = (epoch - warmup_steps) / float(total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        
        lr = lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        if warmup_steps:
            lr = lr * tf.minimum(1.0, epoch / warmup_steps)

        return lr

    return step_fn

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set,loc_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.location = loc_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(idx)
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_loc = self.location[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = [batch_x,batch_loc]
        return x, np.array(batch_y)
    
def get_model(config=config):
    model = MSRLSTM(
        num_classes=8,
        dropout=config.dropout,
        data_dim=config.data_dim,
        # norm_param = config.norm_param,
        batch_size= config.batch_size,
        loc_dim=config.location_dim
    )
    if config.optimizer == "adam":
        optim = Adam(
            global_clipnorm=config.global_clipnorm,
            amsgrad=config.amsgrad,
        )
    elif config.optimizer == "adamw":
        optim = AdamW(
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
            global_clipnorm=config.global_clipnorm,
            exclude_from_weight_decay=["position"]
        )
    else:
        raise ValueError("The used optimizer is not in list of available")

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=optim,
        metrics=["accuracy"],
    )
    return model


def load_weights(model,dir,config=config,epoch=None,earlystopping=True):
    if epoch is not None:
        checkpoint_path = os.path.join(dir,"cp-{epoch:04d}.ckpt").format(epoch=epoch)
        model.load_weights(checkpoint_path)
        return
    files = glob.glob(os.path.join(dir,'*.ckpt.index'))
    patience = 1 if not earlystopping else config.earlystopping_patience
    ckpt = os.path.splitext(files[-patience])[0]
    model.load_weights(ckpt)
    return len(files)-patience