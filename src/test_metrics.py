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
import argparse

import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from keras.models import load_model


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


def plot_metrics(model_dict,metric,x_label,y_label,savepath):
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
    plt.savefig(savepath,format="png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--model', type=str, help="Model to be trained", default="rnn_cnn")
    parser.add_argument('--batch-size', type=int, help="size of batch for training", default=32)
    parser.add_argument('--epochs', type=int, help="num epochs for training", default=15)
    parser.add_argument('--sp-pickles', type=str, help="save folder for all generated pickles", default="../data/pickles")
    parser.add_argument('--sp-models', type=str, help="save folder for all generated models", default="../data/models")
    parser.add_argument('--sp-images', type=str, help="save folder for all graphs", default="../data/images")
    parser.add_argument('--sp-summary', type=str, help="save folder for all generated summary stats", default="../data/sumstats")


    args = parser.parse_args()

    mod_name = args.model
    batch_size = args.batch_size
    num_epochs = args.epochs
    sp_pickles = args.sp_pickles
    sp_models = args.sp_models
    sp_images = args.sp_images
    sp_summary = args.sp_summary


    fn_test = os.path.join(sp_pickles, "test_input.pkl")
    test = pickle.load(open(fn_test, 'rb'))
    X_test = test['X_test']
    y_test = test['y_test']
    docs_test = test['docs_test']

    mod_conf = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '.hdf5'])
    mod_path = os.path.join(sp_models, mod_conf)
    mod_pickle_name = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '.pkl'])
    mod_pickle_path = os.path.join(sp_pickles, mod_pickle_name)
    
    if os.path.isfile(mod_path) and os.path.isfile(mod_pickle_path):    
        mod_hist = pickle.load(open(mod_pickle_path,"rb"))
    else:
        print("Model configuration ", + " doesn't exist. Please try a different configuration or build the model first")
        exit(0)
    # cnn_hist = pickle.load(open("../data/pickles/cnn.pkl","rb"))
    # rnn_hist = pickle.load(open("../data/pickles/rnn.pkl","rb"))
    # rnn_cnn_hist = pickle.load(open("../data/pickles/rnn_cnn.pkl","rb"))


    plt.style.use("ggplot")
    model_dict = {"model": mod_hist}

    img_acc = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '_accuracy.png'])
    ip_acc = os.path.join(sp_images, img_acc)
    plot_metrics(model_dict,"acc","Epoch","Accuracy",ip_acc) ## Accuracy
    img_loss = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '_loss.png'])
    ip_loss = os.path.join(sp_images, img_loss)
    plot_metrics(model_dict,"loss","Epoch","Loss",ip_loss) ## Loss
    img_auc = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '_auc.png'])
    ip_auc = os.path.join(sp_images, img_auc)
    plot_metrics(model_dict,'auc_roc',"Epoch","AUC_ROC",ip_auc) ## AUC_ROC

    mod = load_model(mod_path,custom_objects={"auc_roc":auc_roc})
    
    keys = mod.metrics_names
    val = mod.evaluate([docs_test,X_test],y_test,batch_size=batch_size)

    res_dat = pd.DataFrame('metric': keys, 'value': val)
    res_file = ''.join([mod_name,'_', str(batch_size), '_', str(num_epochs), '.csv'])
    res_path = os.path.join(sp_summary, res_file)
    res.dat(res_path, index = False)
    # cnn.evaluate([docs_test,X_test],y_test,batch_size=64)


    # rnn = load_model("Data/models/rnn.hdf5",custom_objects={"auc_roc":auc_roc})
    # rnn.evaluate([docs_test,X_test],y_test,batch_size=64)


    # rnn_cnn = load_model("Data/models/rnn_cnn.hdf5",custom_objects={"auc_roc":auc_roc})
    # rnn_cnn.evaluate([docs_test,X_test],y_test,batch_size=64)

