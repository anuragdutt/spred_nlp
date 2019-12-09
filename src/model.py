import os
import sys
import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
import gc
import pickle
import pandas

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import concatenate as lconcat
from keras.layers import Dense, Dropout 
from keras.layers import GRU, CuDNNGRU,Input, LSTM, Embedding, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, TimeDistributed, BatchNormalization

#sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils,plot_model, multi_gpu_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
import argparse



def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true,y_pred)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



def build_model(output_classes,architecture,embedding_matrix,aux_shape,vocab_size,embed_dim,max_seq_len,multi_gpu):
    
    with tf.device('/cpu:0'):
        main_input= Input(shape=(max_seq_len,),name='doc_input')
        main = Embedding(input_dim = vocab_size,
                            output_dim = embed_dim,
                            weights=[embedding_matrix], 
                            input_length=max_seq_len, 
                            trainable=False)(main_input)

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        # main = Dense(32, activation='relu')(main)
        main = Dense(32, activation='relu')(main)
        main = Dropout(0.2)(main)
        main = Flatten()(main)
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        main = Conv1D(64, 3, strides=1, padding='same', activation='relu')(main)
        #Cuts the size of the output in half, maxing over every 2 inputs
        main = MaxPooling1D(pool_size=3)(main)
        main = Dropout(0.2)(main)
        main = Conv1D(32, 3, strides=1, padding='same', activation='relu')(main)
        main = GlobalMaxPooling1D()(main)
        #model.add(Dense(output_classes, activation='softmax'))
    elif architecture == 'rnn':
        # LSTM network
        main = Bidirectional(CuDNNGRU(32, return_sequences=False),merge_mode='concat')(main)
        main = BatchNormalization()(main)
    elif architecture =="rnn_cnn":
        main = Conv1D(64, 5, padding='same', activation='relu')(main)
        main = MaxPooling1D()(main)
        main = Dropout(0.2)(main)
        main = Bidirectional(CuDNNGRU(32,return_sequences=False),merge_mode='concat')(main)
        main = BatchNormalization()(main)
   
    else:
        print('Error: Model type not found.')
        
    auxiliary_input = Input(shape=(aux_shape,), name='aux_input')
    x = lconcat([main, auxiliary_input])
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    main_output = Dense(output_classes, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output],name=architecture)
        
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    if multi_gpu:
        model = multi_gpu_model(model)
    model.compile('adam', 'categorical_crossentropy',metrics=['accuracy',auc_roc])
    
    return model
    
def gen():
    print('generator initiated')
    idx = 0
    while True:
        yield [docs_train[:32], X_train[:32]], y_train[:32]
        print('generator yielded a batch %d' % idx)
        idx += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--model', type=str, help="Model to be trained", default="rnn_cnn")
    parser.add_argument('--batch-size', type=int, help="size of batch for training", default=32)
    parser.add_argument('--epochs', type=int, help="num epochs for training", default=15)
    parser.add_argument('--sp-pickles', type=str, help="save folder for all generated pickles", default="../data/pickles")
    parser.add_argument('--sp-models', type=str, help="save folder for all generated models", default="../data/models")
    parser.add_argument('--sp-models', type=str, help="save folder for all generated models", default="../data/models")
    parser.add_argument('--multi-gpu', type=bool, help="Toggle to enable usage of multi-gpu", default=0)


    args = parser.parse_args()

    mod_name = args.model
    batch_size = args.batch_size
    num_epochs = args.epochs
    sp_pickles = args.sp_pickles
    sp_model = args.sp_models
    multi_gpu = args.multi_gpu

    # model_dict = dict()
    config = os.path.join(sp_pickles, 'config.pkl')
    fn_train = os.path.join(sp_pickles, 'train_input.pkl')
    # fn_test = os.path.join(sp_pickles, 'test_input.pkl')

    config = pickle.load(open(config, 'rb'))
    train_dict = pickle.load(open(fn_train, 'rb'))
    # test_dict = pickle.load(open(fn_test))


    embedding_matrix = config['embedding_matrix']
    aux_shape = config['aux_shape']
    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    max_words = config['max_words']    

    docs_train = train_dict['docs_train']
    X_train = train_dict['X_train']
    y_train = train_dict['y_train']

    if mod_name in ["rnn", "cnn", "rnn_cnn"]:
        mod = build_model(3,mod_name, embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words, multi_gpu = multi_gpu)
        print(mod_name + ".......................................................")
        model_fit = mod.fit([docs_train,X_train],y_train,batch_size=batch_size,epochs=num_epochs,verbose=1)

        fn = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '.hdf5'])
        sp = os.path.join(sp_model, fn)

        mod.save(sp)
    
        model_pickle_file = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), ".pkl"])
        model_pickle_path = os.path.join(sp_pickles, model_pickle_file)

        with open(model_pickle_path, 'wb') as file_pi:
            pickle.dump(model_fit.history, file_pi)

    else:
        print("Incorrect model name.................please try again")

    print("Finished training")

    #### Training for more epochs
    # history = rnn_cnn.fit(x = [docs_train,X_train],
    #                   y = y_train,
    #                   batch_size = 32,
    #                   epochs = 20,
    #                   verbose = 1,
    #                   validation_data = ([docs_test,X_test],y_test))