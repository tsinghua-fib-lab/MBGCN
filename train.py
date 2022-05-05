import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from time import time
from tqdm import tqdm
import os
import sys

from loss import bprloss
from dataset import TrainDataset, TestDataset
from utils import EarlyStopManager, ModelSelector, VisManager
from metrics import Recall, NDCG, MRR

BIGNUM = 1e8

class TrainManager(object):
    def __init__(self, flags_obj, vm, cm):
        self.flags_obj = flags_obj
        self.vm = vm
        self.cm = cm
        self.es = EarlyStopManager(flags_obj)
        self.data_set_init(flags_obj)
        self.set_device(flags_obj)
        self.model = ModelSelector.getModel(flags_obj, self.trainset, self.flags_obj.model, self.device).to(self.device)        
        
        self.opt = optim.Adam(self.model.parameters(), lr=flags_obj.lr)
        self.metric_dict = {'Recall10':Recall(10), 'NDCG10':NDCG(10), 'MRR10':MRR(10),
                            'Recall20':Recall(20), 'NDCG20':NDCG(20), 'MRR20':MRR(20),
                            'Recall40':Recall(40), 'NDCG40':NDCG(40), 'MRR40':MRR(40),
                            'Recall80':Recall(80), 'NDCG80':NDCG(80), 'MRR80':MRR(80)}        

    def set_device(self, flags_obj):
        if flags_obj.gpu==True:
            torch.cuda.set_device(flags_obj.gpu_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def data_set_init(self, flags_obj):
        self.trainset = TrainDataset(flags_obj)
        print('Train Data Read Completed!')
        self.validationset = TestDataset(flags_obj, self.trainset,task='validation')
        print('Validation Data Read Completed!')
        self.testset = TestDataset(flags_obj, self.trainset, task='test')
        print('Test Data Read Completed!')
        self.trainloader = DataLoader(self.trainset, flags_obj.batch_size, True, num_workers=flags_obj.num_workers, pin_memory=True)
        self.validationloader = DataLoader(self.validationset, flags_obj.test_batch_size, False, num_workers=flags_obj.num_workers, pin_memory=True)
        self.testloader = DataLoader(self.testset, flags_obj.test_batch_size, False, num_workers=flags_obj.num_workers, pin_memory=True)

    def train(self):
        self.set_leaderboard()

        for epoch in range(self.flags_obj.epoch):
            self.train_one_epoch()
            if self.flags_obj.model=='MBGCN':
                self.multi3_validation()
            else:
                self.validation()
            self.update_leaderboard(epoch)
            self.trainloader.dataset.newit()

            stop = self.es.step(list(self.metric_dict.values())[0]._metric, epoch)
            if stop == True:
                break

    def train_one_epoch(self):
        self.model.train()
        if self.flags_obj.model=='MBGCN':
            print(self.model.mgnn_weight)

        start = time()
        total_loss = 0
        for i, data in enumerate(tqdm(self.trainloader)):
            users, items = data
            self.opt.zero_grad()
            modelout = self.model(users.to(self.device), items.to(self.device))

            loss = bprloss(modelout, batch_size = self.trainloader.batch_size, loss_mode = self.flags_obj.loss_mode)
            total_loss += loss

            loss.backward()
            self.opt.step()

        time_interval = time()-start

        self.vm.update_line('epoch loss', total_loss)
        self.vm.update_line('train time cost', time_interval)

    def validation(self):
        
        self.model.eval()
        for metric in self.metric_dict:
            self.metric_dict[metric].start()
        start = time()
        with torch.no_grad():
            propagate_result = self.model.propagate(task='test')
            for i, data in enumerate(tqdm(self.validationloader)):
                users, ground_truth, train_mask = data
                pred = self.model.evaluate(propagate_result, users.to(self.device))

                pred -= BIGNUM * train_mask.to(self.device)

                for metric in self.metric_dict:
                    self.metric_dict[metric](pred, ground_truth.to(self.device))

        stop = time()
        time_interval = stop - start
        self.vm.update_line('validation time cost', time_interval)

        for metric in self.metric_dict:
            self.metric_dict[metric].stop()
        self.vm.update_metrics(self.metric_dict)
        
        for metric in self.metric_dict:
            print('{}:{}'.format(metric, self.metric_dict[metric]._metric))

    def multi3_validation(self):
        
        self.model.eval()
        for metric in self.metric_dict:
            self.metric_dict[metric].start()
        start = time()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.validationloader)):
                users, ground_truth, train_mask = data
                pred = self.model.evaluate(users.to(self.device))

                pred -= BIGNUM * train_mask.to(self.device)

                for metric in self.metric_dict:
                    self.metric_dict[metric](pred, ground_truth.to(self.device))

        stop = time()
        time_interval = stop - start
        self.vm.update_line('validation time cost', time_interval)

        for metric in self.metric_dict:
            self.metric_dict[metric].stop()
        self.vm.update_metrics(self.metric_dict)
        
        for metric in self.metric_dict:
            print('{}:{}'.format(metric, self.metric_dict[metric]._metric))

    def set_leaderboard(self):

        self.max_metric = -1.0
        self.max_epoch = -1
        self.leaderboard = self.vm.new_text_window('leaderboard')

    def update_leaderboard(self, epoch):

        metric_list = list(self.metric_dict.values())
        metric = metric_list[0]._metric
        if metric > self.max_metric:
            self.max_metric = metric
            self.max_epoch = epoch

            self.vm.append_text('New Record! {} @ epoch {}!'.format(metric, epoch), self.leaderboard)
            self.cm.model_save(self.model)

    def test(self):
        best_model = ModelSelector.getModel(flags_obj, self.trainset, self.flags_obj.model_name)
        self.cm.model_load(best_model)

        model.eval()
        for metric in self.metric_dict:
            metric_dict[metric].start()
        with torch.no_grad():
            propagate_result = self.model.propagate(task='test')
            for users, ground_truth, train_mask in self.testloader:
                pred = self.model.evaluate(propagate_result, users.to(self.device))
                pred -= BIGNUM * train_mask.to(self.device)
                for metric in self.metric_dict:
                    metric_dict[metric](pred, ground_truth.to(self.device))

        for metric in self.metric_dict:
            metric_dict[metric].stop()
        
        self.vm.append_text('Final Test Result:', self.leaderboard)
        for metric in self.metric_dict:
            self.vm.append_text(metric+': {}'.format(self.metric_dict[metric]._metric))
