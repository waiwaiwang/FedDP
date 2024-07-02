import argparse
import logging
import os
import random
import sys

import tqdm
from data_loader_pro import load_partition_data
import numpy as np
import torch

from LR import LogisticRegression
from Fedavg_api_pro import Fedavg
from my_model_trainer_classification_pro import MyModelTrainer as MyModelTrainerCLS
import warnings
warnings.filterwarnings('ignore')

def add_args(parser):
  
    # Training settings
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument("--es_lr", type=float, default=0.0001, metavar="LR", help="learning rate ")

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')
    
    parser.add_argument("--ed_epoch", help="distillation_epoch;", type=int, default=10)

    parser.add_argument('--client_num_in_total', type=int, default=21, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=21, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    
    parser.add_argument("--temperature", help="distillation_T;", type=float, default=5.0)

    parser.add_argument("--p", help="distillation_Sampling_Size;", type=float, default=700)


    return parser

def load_data(args,testname,ra):

    test_data_global,\
    train_data_local_num_dict,\
    train_data_local_dict,\
    test_data_local_dict,\
    class_num,\
    distillation_data = load_partition_data(ra,args.client_num_in_total,testname)

    dataset = [ test_data_global,train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,distillation_data]

    return dataset

def create_model():
    model = LogisticRegression()
    return model

ff = [ 'ant','jedit','lucene', 'xerces'   ,   'velocity','xalan',         'synapse','log4j','poi',       'ivy', 'prop6','redaktor','tomcat']
folder_path = ['ant-1.7','jedit-4.1','lucene-2.4',  'xerces-1.3',          'velocity-1.6','xalan-2.6',            'synapse-1.2','log4j-1.1','poi-3.0',      'ivy-2.0','prop6','redaktor','tomcat']

if __name__ == "__main__":

    auc = []
    accuracy = []
    recall =[]
    precision = []
    f1 = []
    mcc = []

    for i, name in enumerate(ff):

        name = ff[i]
        if i > 9:
            parser.set_defaults(client_num_in_total=22)
            parser.set_defaults(client_num_per_round=22)
            args = parser.parse_args()

        pro_auc = []
        pro_accuracy = []
        pro_recall =[]
        pro_precision = []
        pro_f1 = []
        for ra in range(5):
            parser = add_args(argparse.ArgumentParser(description='Fed'))
            args = parser.parse_args()
            device = torch.device("cpu") 
            epo_auc = []
            epo_accuracy = []
            epo_recall =[]
            epo_precision = []
            epo_f1 = []
            epo_mcc = []
            round_num = 10

            for k in range(round_num):
                round = args.comm_round - k -1
                random.seed(ra)
                np.random.seed(ra)
                torch.manual_seed(ra)
                torch.cuda.manual_seed_all(ra)
                torch.backends.cudnn.deterministic = True

                dataset = load_data(args,name,ra)
                model = create_model()
                model_trainer = MyModelTrainerCLS(model)

                savepath = '...' 

                FedavgAPI = Fedavg(dataset, device, args, model_trainer, savepath,i,ra)
                result = FedavgAPI.test()
                epo_auc.append(result['test_auc']);epo_accuracy.append(result['test_acc']);epo_recall.append(result['test_r']);epo_precision.append(result['test_p']);epo_f1.append(result['test_f1'])
            pro_auc.append(np.mean(epo_auc));pro_accuracy.append(np.mean(epo_accuracy));pro_recall.append(np.mean(epo_recall));pro_precision.append(np.mean(epo_precision));pro_f1.append(np.mean(epo_f1))
        pro_mean_auc = np.mean(pro_auc)
        pro_mean_accuracy = np.mean( pro_accuracy )
        pro_mean_recall =np.mean( pro_recall )
        pro_mean_precision = np.mean( pro_precision )
        pro_mean_f1 = np.mean( pro_f1)
        auc.append(pro_mean_auc);accuracy.append(pro_mean_accuracy );recall.append(pro_mean_recall);precision.append(pro_mean_precision);f1.append(pro_mean_f1)
        print("pro_auc",pro_mean_auc,"pro_accuracy",pro_mean_accuracy,"pro_recall",pro_mean_recall,"pro_precision",pro_mean_precision,"pro_f1",pro_mean_f1)
    mean_auc = np.mean(auc)
    mean_accuracy = np.mean( accuracy )
    mean_recall =np.mean( recall )
    mean_precision = np.mean( precision )
    mean_f1 = np.mean( f1)
    print("auc\n",mean_auc,"\naccuracy\n",mean_accuracy,"\nrecall\n",mean_recall,"\nprecision\n",mean_precision,"\nf1\n",mean_f1)
