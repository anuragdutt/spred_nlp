import os
import sys
import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pickle
import pandas

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
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


global aux_shape
global vocab_size  
global embed_dim
global max_words

def load_embeddings(vec_file):
    print("Loading Glove Model")
    f = open(vec_file,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. {} words loaded!".format(len(model)))
    return model

def tokenize_and_pad(docs,max_words=34603):
    global t
    t = Tokenizer()
    t.fit_on_texts(docs)
    docs = pad_sequences(sequences = t.texts_to_sequences(docs),maxlen = max_words, padding = 'post')
    global vocab_size
    vocab_size = len(t.word_index) + 1
    
    return docs

def oversample(X,docs,y):
    # Get number of rows with imbalanced class
    target = y.sum().idxmax()
    n = y[target].sum()
    # identify imbalanced targets
    imbalanced = y.drop(target,axis=1)
    #For each target, create a dataframe of randomly sampled rows, append to list
    append_list =  [y.loc[y[col]==1].sample(n=n-y[col].sum(),replace=True,random_state=20) for col in imbalanced.columns]
    append_list.append(y)
    y = pd.concat(append_list,axis=0)
    # match y indexes on other inputs
    X = X.loc[y.index]
    docs = pd.DataFrame(docs_train,index=y_train.index).loc[y.index]
    assert (y.index.all() == X.index.all() == docs.index.all())
    return X,docs.values,y

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

def build_model(output_classes,architecture,embedding_matrix,aux_shape,vocab_size,embed_dim,max_seq_len):
    
    with tf.device('/cpu:0'):
        main_input= Input(shape=(max_seq_len,),name='doc_input')
        main = Embedding(input_dim = vocab_size,
                            output_dim = embed_dim,
                            weights=[embedding_matrix], 
                            input_length=max_seq_len, 
                            trainable=False)(main_input)

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
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
    # model = multi_gpu_model(model)
    model.compile('adam', 'categorical_crossentropy',metrics=['accuracy',auc_roc])
    
    return model

def plot_metrics(model_dict,metric,x_label,y_label):
    plots = 1
    plt.figure(figsize=[15,10])
    for model, history in model_dict.items():
        plt.subplot(2,2,plots)
        plt.plot(history[metric])
        #plt.plot(history.history['val_acc'])
        plt.title('{0} {1}'.format(model,metric))
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plots += 1
    #plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig("Graphs/{}.png".format(metric),format="png")
    plt.show()
    
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
    parser.add_argument('--max_words', type=int, help="Max words considered for each 8k filing", default=34603)
    parser.add_argument('--embed_dim', type=int, help="Dimension of Embeddings", default="100")
    parser.add_argument('--batch-size', type=int, help="size of batch for training", default=32)
    parser.add_argument('--epochs', type=int, help="num epochs for training", default=15)
    parser.add_argument('--embed-file', type=str, help="embedding location", default='../data/glove/glove.6B.100d.txt')
    parser.add_argument('--filing-file', type=str, help="data for all sec filings", default="../data/embedded_data/sample_data.csv.gz")
    parser.add_argument('--sp-summary', type=str, help="save folder for all generated summary stats", default="../data/sumstats")
    parser.add_argument('--sp-pickles', type=str, help="save folder for all generated pickles", default="../data/pickles")
    parser.add_argument('--sp-models', type=str, help="save folder for all generated models", default="../data/models")


    args = parser.parse_args()

    mod_name = args.model
    max_words = args.max_words
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    num_epochs = args.epochs
    embed_file = args.embed_file
    filing_file = args.filing_file
    sp_summary = args.sp_summary
    sp_pickles = args.sp_pickles
    sp_model = args.sp_models

    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
    cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
    cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")

    cik_df = cik_df.loc[:, ['CIK', 'GICS Sector', 'GICS Sub Industry']]
    cik_df.columns = ['cik', 'GICS Sector', 'GICS Sub Industry']




    # df = pd.read_pickle("../data/pickles/lemmatized_data.pickle")
    # print("******************************************************")

    df = pd.read_csv(filing_file, compression = "gzip")
    # df = df.loc[:200]
    # df.to_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip", index = False)
    # df = pd.read_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip")
    # exit(0)

    df = df.dropna()
    print(df.shape)
    # exit(0)

    df = pd.merge(df, cik_df, on = "cik", how = "left")


    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('items')),columns=mlb.classes_,),sort=False,how="left")
    df_sumstats_path = os.path.join(sp_summary, "main_data.csv.gz")
    df.to_csv(df_sumstats_path, compression = "gzip")

    cols = ['GICS Sector','vix','rm_week','rm_month', 'rm_qtr', 'rm_year']
    cols.extend(list(mlb.classes_))
    X = df[cols]
    docs = df['processed_text']
    y = df['signal']    


    # Get Dummies

    docs = tokenize_and_pad(docs)
    X = pd.get_dummies(columns = ['GICS Sector'],prefix="sector",data=X)
    y = pd.get_dummies(columns=['signal'],data=y)


    aux_shape = len(X.columns)
    X_train, X_test, y_train, y_test, docs_train, docs_test = train_test_split(X, y,docs,
                                                    stratify=y, 
                                                    test_size=0.3,
                                                    random_state = 20)


    cont_features = ['vix','rm_week','rm_month', 'rm_qtr', 'rm_year']
    aux_features = cont_features + [item for item in mlb.classes_]
    x_scaler = StandardScaler()
    X_train[cont_features] = x_scaler.fit_transform(X_train[cont_features])
    X_test[cont_features] = x_scaler.transform(X_test[cont_features])

    X_train, docs_train, y_train = oversample(X_train, docs_train, y_train)
    embeddings_index = load_embeddings(embed_file)
    words_not_found = []


    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



    # Save data
    fn_docs_train = os.path.join(sp_pickles, "docs_train.npy")
    fn_docs_test = os.path.join(sp_pickles, "docs_test.npy")

    np.save(fn_docs_train, docs_train)
    np.save(fn_docs_test,docs_test)

    fn_x_train = os.path.join(sp_pickles, "X_train.pkl")
    fn_x_test = os.path.join(sp_pickles, "X_test.pkl")

    X_train.to_pickle(fn_x_train)
    X_test.to_pickle(fn_x_test)

    fn_y_train = os.path.join(sp_pickles, "y_train.pkl")
    fn_y_test = os.path.join(sp_pickles, "y_test.pkl")

    y_train.to_pickle(fn_y_train)
    y_test.to_pickle(fn_y_train) 

    embedding_path = os.path.join(sp_pickles, "embedding_matrix.npy")
    np.save(embedding_path,embedding_matrix)

    model_dict = dict()

    if mod_name in ["rnn", "cnn", "rnn_cnn"]:
        mod = build_model(3,mod_name, embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words)
        print(mod_name + ".......................................................")
        model_fit = mod.fit([docs_train,X_train],y_train,batch_size=batch_size,epochs=num_epochs,verbose=1)

        fn = ''.join([mod_name,'_', batch_size, '_', num_epochs, '.hdf5'])
        sp = os.path.join(save_path, fn)

        mod.save(sp)
    
        model_pickle_file = ''.join([mod_name, ".pkl"])
        model_pickle_path = os.path.join(sp_pickles, model_pickle_file)

        with open(model_pickle_path, 'wb') as file_pi:
            pickle.dump(model_fit, file_pi)

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