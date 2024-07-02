import logging
import random
import traceback
import copy

from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from data_batch import MyDataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model_trainer import ModelTrainer
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    

    def get_probability_density(self,train_data,distillation_share_data, device ,args):
        dis_x,dis_y = distillation_share_data
        train_dataset = torch.load(train_data)
        x_train_resampled = train_dataset["x_train_list"]
        x_train_resampled =  x_train_resampled.float()
        similarities = []
        for dis in dis_x:
            dis =dis.float()
            similarity = F.cosine_similarity(x_train_resampled, dis.unsqueeze(0), dim=1)
            similarities.append(similarity)
        mean_similarities = torch.mean(torch.stack(similarities), dim=1)
        concatenated_pd =mean_similarities.float()

        return concatenated_pd
    

    def get_pd_log(self,client_indexes , device, batch_ds_data, model_dict, pd_dict,num):
        tmp_model = copy.deepcopy(self.model)
        tmp_model.to(device)
        x = batch_ds_data
        x=x.to(device)
        pd_log = None

        with torch.no_grad():
            for idx in client_indexes:
                tmp_model.load_state_dict(model_dict[idx])
                tmp_log = tmp_model(x).to('cpu') 

                for i in range(len(tmp_log)):
                    tmp_log[i] = tmp_log[i] * pd_dict[idx][num+i]
                if pd_log == None:
                    pd_log = tmp_log
                else:
                    pd_log += tmp_log
        return pd_log

    def weighted_ensemble_distillation(self, args, device, global_ds_dataset, client_indexes, model_dict, pd_dataset_dict,
                                    avg_model_params,ra):

        print("################weighted_ensemble_distillation################")
        
        T = args.temperature
        epoch_loss = []
        teacher_acc = []
        student_acc = []
        model_base = self.model
        model_base.load_state_dict(avg_model_params)

        model_base.to(device)
        d_optimizer = torch.optim.AdamW( self.model.parameters(), lr=args.es_lr)
        criterion = nn.KLDivLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer,
                                                        mode="min",
                                                        factor=0.2,
                                                        patience=1)

        for epoch in range(args.ed_epoch):
            x, y = global_ds_dataset    
            dataset = MyDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size)
            pos = 0
            for batch_x, batch_y in dataloader:
                pd_dict = dict()
                pd_dict = pd_dataset_dict  
                
                teacher_log = self.get_pd_log(client_indexes, device,batch_x, model_dict, pd_dict,pos).to(device)
                pos+=len(batch_x)
                batch_x = batch_x.to(device)
                d_optimizer.zero_grad()
                avg_log = model_base(batch_x)
                loss = (T ** 2) * criterion(
                        torch.nn.functional.log_softmax(avg_log / T, dim=1),
                        torch.nn.functional.softmax(teacher_log / T, dim=1)
                    )
                loss.backward()
                d_optimizer.step()
                epoch_loss.append(loss.item())
            scheduler.step()

        print('Epoch: {}\tED_Loss: {:.6f}'.format(epoch, sum(epoch_loss) / len(epoch_loss)))
            
           
        return self.get_model_params()
            

    def train(self, train_data, device, args, test_data,ra):

        train_dataset = torch.load(train_data+'_sample.pt')
        x_train_resampled = train_dataset["x_train_list"]
        y_train_resampled = train_dataset[ "y_train_list"]

        num = len(x_train_resampled)

        model = self.model
        model.to(device)
        model.float()
        
        optimizer = optim.AdamW( self.model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.2)
        cuda = next(model.parameters()).device

        for epoch in range(args.epochs):
            model.train()
            dataset = MyDataset(x_train_resampled, y_train_resampled)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            train_loss = 0.0
            train_acc =0
            train_correct = 0
            train_total = 0
            for batch_x, batch_y in dataloader:

                batch_x.to(cuda)
                batch_x = batch_x.cuda(device)
                batch_y.to(cuda)
                batch_y=batch_y.cuda(device)


                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.float(),batch_y.long())

                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)               

                predicted.to(cuda)
                predicted=predicted.cuda(device)

                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                train_loss += loss.item() * batch_x.size(0)
            train_acc = train_correct / train_total
            train_loss = train_loss/num
            scheduler.step()
        return num

    def test_traindata(self, test_data, device, args,ra):
        test_dataset = torch.load(test_data)

        x_test_pad_resampled = test_dataset["x_train_list"]
        y_test_resampled = test_dataset[ "y_train_list"]

        num = len(x_test_pad_resampled)

        model = self.model
        model.to(device)
        model.float()
        cuda = next(model.parameters()).device 

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'test_tn': 0,
            'test_fp': 0,
            'test_fn': 0,
            'test_tp': 0
        }

        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            dataset = MyDataset(x_test_pad_resampled, y_test_resampled)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_tn = 0
            test_fp = 0
            test_fn = 0
            test_tp = 0

            for batch_x, batch_y in dataloader:
                batch_x.to(cuda)
                batch_x = batch_x.cuda(device)
                batch_y.to(cuda)
                batch_y=batch_y.cuda(device)

                outputs = model(batch_x)
                loss = criterion(outputs.float(),batch_y.long())
                _, predicted = torch.max(outputs, 1)               

                predicted.to(cuda)
                predicted=predicted.cuda(device)

                test_loss += loss.item() * batch_x.size(0)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                cm = confusion_matrix(batch_y.cpu().numpy(), predicted.cpu().numpy(), labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                test_tn += tn
                test_fp += fp
                test_fn += fn
                test_tp += tp
            
            metrics['test_correct'] = test_correct
            metrics['test_loss'] = test_loss
            metrics['test_total'] = test_total
            metrics['test_tn'] =test_tn
            metrics['test_fp'] =test_fp
            metrics['test_fn'] =test_fn
            metrics['test_tp'] =test_tp
        return metrics
    
    def test(self, test_data, device, args,ra):

        test_dataset = pd.read_csv(test_data)
        x_test = test_dataset.iloc[:, 3:-1] 
        y_test = test_dataset.iloc[:, -1]
        x_test = torch.tensor(x_test.values)
        y_test = torch.tensor(y_test.values)

        model = self.model
        model.to(device)
        model.float()
        cuda = next(model.parameters()).device 

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'test_f1': 0,
            'test_auc':0,
            'test_p': 0,
            'test_r': 0,
            'test_acc': 0
        }

        model.eval()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            dataset = MyDataset(x_test, y_test)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_tn = 0
            test_fp = 0
            test_fn = 0
            test_tp = 0
            test_prob_all = []
            test_lable_all = []

            for batch_x, batch_y in dataloader:

                batch_x.to(cuda)
                batch_x = batch_x.cuda(device)
                batch_y.to(cuda)
                batch_y=batch_y.cuda(device)

                outputs = model(batch_x)
                loss = criterion(outputs.float(),batch_y.long())
                _, predicted = torch.max(outputs, 1)               

                predicted.to(cuda)
                predicted=predicted.cuda(device)

                test_loss += loss.item() * batch_x.size(0)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                cm = confusion_matrix(batch_y.cpu().numpy(), predicted.cpu().numpy(), labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                test_tn += tn
                test_fp += fp
                test_fn += fn
                test_tp += tp

                test_prob_all.extend(predicted.cpu().tolist())
                test_lable_all.extend(batch_y.cpu().tolist())
            test_f1 = f1_score(test_lable_all,test_prob_all)
            test_auc =roc_auc_score(test_lable_all, test_prob_all)
            test_p = test_tp/(test_tp+test_fp)
            test_r = test_tp/(test_tp+test_fn)
            test_acc = test_correct /test_total

            metrics['test_correct'] = test_correct
            metrics['test_loss'] = test_loss/len(dataloader.dataset)
            metrics['test_total'] = test_total
            metrics['test_f1'] =test_f1
            metrics['test_auc'] = test_auc
            metrics['test_p'] =test_p
            metrics['test_r'] = test_r
            metrics['test_acc'] = test_acc
        return metrics
