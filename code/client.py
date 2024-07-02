
import numpy as np
import torch.nn.functional as F

class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.probability_density = dict()
        
    def get_sample_number(self):
        return self.local_sample_number
    def set_sample_number(self,num):
        self.local_sample_number = num

    def train(self,w_global,distillation_share_data,ra):#
        self.model_trainer.id = self.client_idx
        self.model_trainer.set_model_params(w_global)
        num = self.model_trainer.train(self.local_training_data, self.device, self.args,self.local_test_data,ra)
        weights = self.model_trainer.get_model_params()
        self.set_sample_number(num)
        probability_density_d = self.model_trainer.get_probability_density(self.local_training_data,distillation_share_data,self.device,self.args)#
        return weights,probability_density_d #
    def local_test(self, b_use_test_dataset,ra):
        if b_use_test_dataset == 1 :
            test_data = self.local_test_data  
            metrics = self.model_trainer.test(test_data, self.device, self.args,ra)        
        else:
            test_data = self.local_training_data 
            metrics = self.model_trainer.test_traindata(test_data, self.device, self.args,ra)
        return metrics
    
    


