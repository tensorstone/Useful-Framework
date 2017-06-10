# -*- coding: utf-8 -*-
"""
This module defines the neural network model,
mainly based on keras.
"""

# Author: evanzd <dongzhou@pku.edu.cn>

from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, LSTM, Flatten
from keras.layers import concatenate
from keras.optimizers import RMSprop


def build_model(loss='binary_crossentropy'):
    """ define neural network model for stock prediction.
    the model is fixed for this single task purpose only.

    Parameters
    ----------
    loss : binary_crossentropy for binary classification and mse for regression

    Returns
    ----------
    model : a compiled keras model
    """
    # check loss
    if loss not in ['binary_crossentropy', 'mse']:
        raise NotImplementedError('loss {} not implemented'.format(loss))
    if loss == 'binary_crossentropy':
        output_activation = 'sigmoid'
        metrics = ['accuracy']
    else:
        output_activation = 'linear'
        metrics = []
    # price
    b1_input = Input(shape=(30,4), name='price')
    b1_lstm  = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(b1_input)
    # volume
    b2_input = Input(shape=(30,1), name='volume')
    b2_lstm  = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(b2_input)
    # high_low_ma
    b3_input = Input(shape=(3,3,1), name='max_min_ma')
    b3_conv  = Conv2D(32, (2,2), padding='same')(b3_input)
    b3_act   = Activation('tanh')(b3_conv)
    b3_fc    = Flatten()(b3_conv) # 288
    # vola_pos_slope
    b4_input = Input(shape=(3,1,3), name='vola_pos_slope')
    b4_conv  = Conv2D(32, (2,1), padding='same')(b4_input)
    b4_act   = Activation('tanh')(b4_conv)
    b4_fc    = Flatten()(b4_conv) # 288
    # clse_shr
    b5_input = Input(shape=(2,), name='clse_shr')
    # merge
    merged  = concatenate([b1_lstm, b2_lstm, b3_fc, b4_fc, b5_input])
    # trunk
    t1 = Dense(512)(merged)
    t2 = Activation('tanh')(t1)
    t3 = Dropout(0.25)(t2)
    t4 = Dense(512)(t3)
    t5 = Activation('tanh')(t4)
    t6 = Dropout(0.25)(t5)
    t7 = Dense(512)(t6)
    t8 = Activation('tanh')(t7)
    t9 = Dropout(0.25)(t8)
    t10 = Dense(1)(t9)
    output = Activation(output_activation)(t10)
    # model
    model = Model(inputs=[b1_input, b2_input, b3_input, b4_input, b5_input], outputs=output)
    optimizer = RMSprop(lr=1e-3)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    return model

