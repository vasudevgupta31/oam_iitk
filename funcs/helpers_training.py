import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


class SeqModel():
    """Class to define the language model, i.e the neural net"""
    def __init__(self, n_chars, max_length, layers, dropouts, trainables, lr, verbose=False):  
        
        self.n_chars = n_chars
        self.max_length = max_length
        
        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))
        
        for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
            self.model.add(LSTM(neurons, 
                                unit_forget_bias=True, 
                                dropout=dropout, 
                                trainable=trainable, 
                                return_sequences=True))        
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, activation='softmax')))

        optimizer = Adam(learning_rate=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def create_model_checkpoint(period, save_path):
    """ Function to save the trained model during training """
    filepath = os.path.join(save_path, 'epoch_{epoch:02d}.h5')
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=0,
                                   save_best_only=False,
                                   save_freq=period)

    return checkpointer
