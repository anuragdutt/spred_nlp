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
    model = multi_gpu_model(model)
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

if __name__ == "__main__":

	max_words = 34603
	embed_dim = 100

	wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
	cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
	cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
	cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")

	cik_df = cik_df.loc[:, ['CIK', 'GICS Sector', 'GICS Sub Industry']]
	cik_df.columns = ['cik', 'GICS Sector', 'GICS Sub Industry']

	# df = pd.read_pickle("../data/pickles/lemmatized_data.pickle")
	# print("******************************************************")

	df = pd.read_csv("../data/embedded_data/main_data_3.csv.gz", compression = "gzip")
	# df = df.loc[:2000]
	# df.to_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip", index = False)
	# df = pd.read_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip")
	df = df.dropna()
	print(df.shape)
	# exit(0)

	df = pd.merge(df, cik_df, on = "cik", how = "left")


	mlb = MultiLabelBinarizer()
	df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('items')),columns=mlb.classes_,),sort=False,how="left")
	
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
	embeddings_index = load_embeddings("../data/glove/glove.6B.100d.txt")
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
	np.save("../data/pickles/docs_train.npy",docs_train)
	np.save("../data/pickles/docs_test.npy",docs_test)

	X_train.to_pickle("../data/pickles/X_train.pkl")
	X_test.to_pickle("../data/pickles/X_test.pkl")

	y_train.to_pickle("../data/pickles/y_train.pkl")
	y_test.to_pickle("../data/pickles/y_test.pkl") 

	np.save("../data/pickles/embedding_matrix.npy",embedding_matrix)

	model_dict = dict()


	mlp = build_model(3,"mlp", embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words)
	print("mlp.......................................................")
	model_dict["mlp"] = mlp.fit([docs_train,X_train],y_train,batch_size=32,epochs=10,verbose=1) 

	mlp.save("../data/models/mlp.hdf5")
	# with open('../data/train_history/mlp.pkl', 'wb') as file_pi:
	    # pickle.dump(model_dict["mlp"], file_pi)



	rnn = build_model(3,"rnn", embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words)
	print("rnn.......................................................")
	model_dict["rnn"] = rnn.fit([docs_train,X_train],y_train,batch_size=32,epochs=10,verbose=1)

	rnn.save("../data/models/rnn.hdf5")
	# with open('../data/train_history/rnn.pkl', 'wb') as file_pi:
	#     pickle.dump(model_dict["rnn"].history, file_pi)

	cnn = build_model(3,"cnn", embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words)
	print("cnn.......................................................")
	model_dict["cnn"] = cnn.fit([docs_train,X_train],y_train,batch_size=32,epochs=10,verbose=1)

	cnn.save("../data/models/cnn.hdf5")
	# with open('..d/data/train_history/cnn.pkl', 'wb') as file_pi:
	#     pickle.dump(model_dict["cnn"].history, file_pi)



	rnn_cnn = build_model(3,"rnn_cnn", embedding_matrix = embedding_matrix, aux_shape = aux_shape, vocab_size = vocab_size, embed_dim = embed_dim, max_seq_len = max_words)
	print("rnn_cnn.......................................................")
	model_dict["rnn_cnn"] = rnn_cnn.fit([docs_train,X_train],y_train,batch_size=32,epochs=10,verbose=1)

	rnn_cnn.save("../data/models/rnn_cnn.hdf5")
	# with open('Data/train_history/rnn_cnn.pkl', 'wb') as file_pi:
	#     pickle.dump(model_dict["rnn_cnn"].history, file_pi)


	exit(0)



	X_test = pd.read_pickle("../data/pickles/X_test.pkl")
	y_test = pd.read_pickle("../data/pickles/y_test.pkl")
	docs_test = np.load("../data/pickles/docs_test.npy")

	mlp_hist = pickle.load(open("../data/pata/trainHistory/mlp.pkl","rb"))
	cnn_hist = pickle.load(open("../data/pata/trainHistory/cnn.pkl","rb"))
	rnn_hist = pickle.load(open("../data/pata/trainHistory/rnn.pkl","rb"))
	rnn_cnn_hist = pickle.load(open("../data/pata/trainHistory/rnn_cnn.pkl","rb"))


	plt.style.use("ggplot")
	model_dict = {"mlp": mlp_hist,
	              "cnn": cnn_hist,
	              "rnn": rnn_hist,
	              "rnn_cnn": rnn_cnn_hist}

	plot_metrics(model_dict,"acc","Epoch","Accuracy") ## Accuracy
	plot_metrics(model_dict,"loss","Epoch","Loss") ## Loss
	plot_metrics(model_dict,'auc_roc',"Epoch","AUC_ROC") ## AUC_ROC


	mlp.evaluate([docs_test,X_test],y_test,batch_size=64)
	cnn.evaluate([docs_test,X_test],y_test,batch_size=64)


	rnn = load_model("Data/models/rnn.hdf5",custom_objects={"auc_roc":auc_roc})
	rnn.evaluate([docs_test,X_test],y_test,batch_size=64)


	rnn_cnn = load_model("Data/models/rnn_cnn.hdf5",custom_objects={"auc_roc":auc_roc})
	rnn_cnn.evaluate([docs_test,X_test],y_test,batch_size=64)


	#### Training for more epochs
	history = rnn_cnn.fit(x = [docs_train,X_train],
	                  y = y_train,
	                  batch_size = 32,
	                  epochs = 20,
	                  verbose = 1,
	                  validation_data = ([docs_test,X_test],y_test))