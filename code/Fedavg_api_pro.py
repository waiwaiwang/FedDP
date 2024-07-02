import copy
import logging
import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from utils import transform_list_to_tensor
from client import Client

ff = [ 'ant','jedit','lucene', 'xerces'   ,   'velocity','xalan',         'synapse','log4j','poi',       'ivy', 'prop6','redaktor','tomcat']
folder_path = ['ant-1.7','jedit-4.1','lucene-2.4',  'xerces-1.3',          'velocity-1.6','xalan-2.6',            'synapse-1.2','log4j-1.1','poi-3.0',      'ivy-2.0','prop6','redaktor','tomcat']

class Fedavg(object):
    def __init__(self, dataset, device, args, model_trainer, savepath,proid,ra):
        self.proid = proid
        self.ra = ra 
        self.device = device
        self.args = args
        self.savepath = savepath

        [test_data_global,local_num_dict, train_data_local_dict, test_data_local_dict,_,distillation_data] = dataset ###
        self.client_indexes = [] 
        self.client_list = []  
        self.train_data_local_num_dict = local_num_dict 
        self.train_data_local_dict = train_data_local_dict 
        self.test_data_local_dict = test_data_local_dict 
        self.test_data_global = test_data_global
        self.model_dict = dict()  
        self.model_trainer = model_trainer
        self.distillation_data = distillation_data
        self.probability_density_dict = dict()        
        self.dis_resample = None

        self.train_acc = []
        self.test_acc = []
        self.train_loss = []
        self.test_loss = []
        self.train_f1 = []
        self.test_f1 = []

        self._setup_clients(local_num_dict, train_data_local_dict, test_data_local_dict)  

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):

        for client_idx in range(self.args.client_num_in_total):

            c = Client(client_idx, train_data_local_dict[client_idx],test_data_local_dict[client_idx],train_data_local_num_dict[client_idx], self.args,self.device,self.model_trainer)
            self.client_list.append(c)

    def train(self):

        for round_idx in range(self.args.comm_round): 
            w_global = self.model_trainer.get_model_params() 
            print("################Communication round : {}".format(round_idx))
            
            self._generate_validation_set(self.args.p) 
            w_locals = [] 

            self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            print("client_indexes = " + str(self.client_indexes))

            print("-------model actually train------")

            for idx in self.client_indexes:
                client = self.client_list[idx]
                weight,probability_density= client.train(copy.deepcopy(w_global),self.dis_resample,self.ra)
                w_locals.append((client.get_sample_number(), copy.deepcopy(weight)))
                for i in range(len(probability_density)):
                    probability_density[i] = probability_density[i].cpu().detach().tolist()

                self.probability_density_dict[client.client_idx] = probability_density

                self.model_dict[client.client_idx] = transform_list_to_tensor(copy.deepcopy(weight))

            probability_density_sum = torch.zeros_like(self.probability_density_dict[self.client_indexes[0]])

            for idx in self.client_indexes:
                probability_density_sum += self.probability_density_dict[idx]

            for idx in self.client_indexes:
                for i in range(len(self.probability_density_dict[idx])):
                    if probability_density_sum[i] !=0 :
                        self.probability_density_dict[idx][i]= self.probability_density_dict[idx][i] / probability_density_sum[i]
                    else:
                        self.probability_density_dict[idx][i] = 0

            w_global = self._aggregate(w_locals,round_idx)
           
            if round_idx % 1 == 0:
                self._local_test_on_all_clients(round_idx)
                if round_idx > (self.args.comm_round-11):
                    pa = self.model_trainer.get_model_params()
                    torch.save(pa,'...')

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            self.client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  
            self.client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

    def _generate_validation_set(self, num_samples):
        num = num_samples
        ds_dataset = torch.load(self.distillation_data+'.pt')

        x_train_pad =ds_dataset["x_train_list"]
        y_train = ds_dataset[ "y_train_list"]

        data_allnum = len(x_train_pad)
        sample_indices = random.sample(range(data_allnum), min(num, data_allnum))
        x_sub = torch.utils.data.Subset(x_train_pad, sample_indices)
        y_sub = torch.utils.data.Subset(y_train, sample_indices)
        self.dis_resample = (x_sub,y_sub)


    def _aggregate(self, w_locals,round_idx):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num,averaged_params) = w_locals[idx]
            training_num += sample_num
        print("###  aggregate : {}".format(len(w_locals)))

        (sample_num,averaged_params) = w_locals[0]
        for k in averaged_params.keys():##############
            for i in range(0, len(w_locals)):
                local_sample_number,local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w 

        global_model_params = averaged_params
        global_model_params = self.model_trainer.weighted_ensemble_distillation(
                self.args,self.device,
                self.dis_resample, self.client_indexes,
                self.model_dict, self.probability_density_dict,
                 global_model_params,self.ra
            )

        self.model_trainer.set_model_params(global_model_params)

        return global_model_params

    def _local_test_on_all_clients(self, round_idx):

        print("###  local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': [],
            'tn': [],
            'fp':[],
            'fn': [],
            'tp':[]
        }

        for client in self.client_list:
            train_local_metrics = client.local_test(0,self.ra)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))
            train_metrics['tn'].append(copy.deepcopy(train_local_metrics['test_tn']))
            train_metrics['fp'].append(copy.deepcopy(train_local_metrics['test_fp']))
            train_metrics['fn'].append(copy.deepcopy(train_local_metrics['test_fn']))
            train_metrics['tp'].append(copy.deepcopy(train_local_metrics['test_tp']))

        client = self.client_list[1]
        test_local_metrics = client.local_test(1,self.ra)
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_p = sum(train_metrics['tp'])/(sum(train_metrics['tp'])+sum(train_metrics['fp']))
        train_r = sum(train_metrics['tp'])/(sum(train_metrics['tp'])+sum(train_metrics['fn']))
        train_f1 =2.0*train_p*train_r/(train_r+train_p)
        
        test_acc =test_local_metrics['test_acc']
        test_loss = test_local_metrics['test_loss']
        test_f1=test_local_metrics['test_f1']
        test_auc=test_local_metrics['test_auc']
        test_p=test_local_metrics['test_p']
        test_r=test_local_metrics['test_r']


        stats = {'training_loss': train_loss,'training_acc': train_acc, 'training_f1':train_f1}
        formatted_stats = {k: f'{v:.6f}' for k, v in stats.items()}
        print(formatted_stats)

        stats = {'test_loss': test_loss,'test_acc': test_acc, 'test_f1':test_f1,'test_auc': test_auc,'test_r':test_r}
        formatted_stats = {k: f'{v:.6f}' for k, v in stats.items()}
        print(formatted_stats)
        
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)

        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)

        return train_f1,train_loss
    
    def test(self):

        model_pa = torch.load(self.savepath)
        self.model_trainer.set_model_params(model_pa)
        client = Client(0, 'none',self.test_data_global,0, self.args,self.device,self.model_trainer)
        test_local_metrics = client.local_test(1,self.ra)
        test_acc =test_local_metrics['test_acc']
        test_loss = test_local_metrics['test_loss']
        test_f1=test_local_metrics['test_f1']
        test_auc=test_local_metrics['test_auc']
        test_p=test_local_metrics['test_p']
        test_r=test_local_metrics['test_r']

        stats = {'test_acc': test_acc, 'test_f1':test_f1,'test_auc': test_auc,'test_p':test_p,'test_r':test_r}
        formatted_stats = {k: f'{v:.6f}' for k, v in stats.items()}
        return stats





