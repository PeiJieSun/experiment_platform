'''
    This is the variant of bpr, which modify loss function with CrossEntropyLoss, 
    in order to output the probability of each rating, like -log(p(r_ai))
'''

import torch
import torch.nn as nn
import random
import string

class bpr(nn.Module):
    def __init__(self, conf):
        super(bpr, self).__init__()
        self.conf = conf
        self.initializeNodes()
        self.defineMap()
    
    def setDevice(self, device):
        self.device = device
    
    def tensorToScalar(self, tensor):
        return tensor.cpu().detach().numpy()
        
    def initializeNodes(self):
        torch.manual_seed(0)
        self.user_embedding = nn.Embedding(self.conf.num_users, self.conf.dimension)
        torch.manual_seed(0)
        self.item_embedding = nn.Embedding(self.conf.num_items, self.conf.dimension)
        # the number 5 here denotes there are five kinds of ratings
        '''NOTICE: Should we add some non-linear function here ?'''
        torch.manual_seed(0)
        self.rating_mapping_layer = nn.Linear(self.conf.dimension, 5)
    
    def initOptimizer(self, model):
        if 'criterion_weight_flag' in self.conf.conf_dict:
            print('setting criterion_weight_flag test pass')
            self.criterion = nn.CrossEntropyLoss()
        else:   
            self.CrossEntropyLoss_weight = torch.Tensor(self.conf.criterion_weight).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.CrossEntropyLoss_weight)
        self.output_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.conf.learning_rate)

    def train(self, feed_dict):
        #self.zero_grad()
        user_input = torch.LongTensor(feed_dict['USER_INPUT']).to(self.device)
        item_input = torch.LongTensor(feed_dict['ITEM_INPUT']).to(self.device)
        label_input = torch.LongTensor(feed_dict['RATING_INPUT']).to(self.device)

        user_vector = self.user_embedding(user_input) #size: batch * self.conf.dimension
        item_vector = self.item_embedding(item_input) #size: batch * self.conf.dimension
        mul_vector = user_vector * item_vector #size: batch * self.conf.dimension
        prediction = torch.sigmoid(self.rating_mapping_layer(mul_vector)) #size: batch * 5
        
        # the self.opt_loss here is the corss entropy loss, Negative Like-lihood Loss -p(\hat{r_ai})
        self.opt_loss = self.criterion(prediction, label_input) # 1 * 1
        self.output_loss = self.output_criterion(prediction, label_input)
        #self.opt_loss.backward()
        #self.optimizer.step()
        self.defineTrainOutMap()
    
    def optim(self):
        self.zero_grad()
        self.opt_loss.backward()
        self.optimizer.step()
    
    def defineMap(self):
        map_dict = {}

        map_dict['train'] = {
            'USER_INPUT': 'USER_LIST', 
            'ITEM_INPUT': 'ITEM_LIST',
            'RATING_INPUT': 'RATING_LIST'
        }

        self.map_dict = map_dict
    
    def defineTrainOutMap(self):
        self.map_dict['out'] = {
            #'output_loss': self.tensorToScalar(self.opt_loss)
            'output_loss': self.tensorToScalar(self.output_loss)
        }