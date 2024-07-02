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
    for ra in range(5):
        parser = add_args(argparse.ArgumentParser(description='Fed'))
        args = parser.parse_args()
        device = torch.device("...") 

        for i, name in tqdm.tqdm(enumerate(ff),desc='project: '):
            print("-------------  {}  ---------------".format(name))        
            if i > 9:
                parser.set_defaults(client_num_in_total=22)
                parser.set_defaults(client_num_per_round=22)
                args = parser.parse_args()
            dataset = load_data(args,name,ra)
            model = create_model()
            model_trainer = MyModelTrainerCLS(model)
            savepath = '...'
            FedavgAPI = Fedavg(dataset, device, args, model_trainer, savepath,i,ra)
            FedavgAPI.train()
            FedavgAPI.draw()


