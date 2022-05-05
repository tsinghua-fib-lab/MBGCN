import visdom
import torch
import numpy as np

import random
import setproctitle
import os
import sys

from absl import logging
from absl import flags
from visdom import Visdom

from model import MF, MBGCN
from dataset import TrainDataset

class ContentManager(object):
    def __init__(self, flag_obj):
        self.name = flag_obj.name
        self.dataset_name = flag_obj.dataset_name
        self.output_path = flag_obj.output
        self.path = os.path.join(self.output_path, self.dataset_name, self.name)
        self.set_proctitle()
        self.output_init()

    def set_proctitle(self):
        setproctitle.setproctitle(self.name)

    def output_init(self):
        if not os.path.exists(os.path.join(self.output_path, self.dataset_name)):
            os.mkdir(os.path.join(self.output_path, self.dataset_name))

        if not os.path.exists(os.path.join(self.output_path, self.dataset_name, self.name)):
            os.mkdir(os.path.join(self.output_path, self.dataset_name, self.name))

    def model_save(self, model):
        torch.save(model.state_dict(), os.path.join(self.path, 'model.pkl'))

    def model_load(self, model):
        model.load_state_dict(torch.load(os.path.join(self.path, 'model.pkl')))


class VisManager(object):
    def __init__(self,flag_obj):
        self.name = flag_obj.name + '_vm'
        self.exp_name = flag_obj.name
        self.__get_port(flag_obj)
        self.set_visdom()
        self.show_basic_info(flag_obj)

    def __get_port(self,flag_obj):
        self.port = flag_obj.port

    def set_visdom(self):

        self.vis = Visdom(port=self.port, env = self.exp_name)

    def show_basic_info(self, flag_obj):

        basic = self.vis.text('Basic Information:')
        self.vis.text('Name: {}'.format(flag_obj.name),win=basic,append=True)
        self.vis.text('Model: {}'.format(flag_obj.model), win=basic, append=True)
        self.vis.text('Dataset: {}'.format(flag_obj.dataset_name), win=basic, append=True)
        self.vis.text('Embedding Size: {}'.format(flag_obj.embedding_size), win=basic, append=True)
        self.vis.text('Node dropout: {}'.format(flag_obj.node_dropout), win=basic, append=True)
        self.vis.text('Message dropout: {}'.format(flag_obj.message_dropout), win=basic, append=True)
        self.vis.text('Initial lr: {}'.format(flag_obj.lr), win=basic, append=True)
        self.vis.text('Batch size: {}'.format(flag_obj.batch_size), win=basic, append=True)
        self.vis.text('Early stop patience: {}'.format(flag_obj.es_patience), win=basic, append=True)
        self.vis.text('L2_norm: {}'.format(flag_obj.L2_norm), win=basic, append=True)

        self.basic = basic

    def update_line(self, title, value):
        
        if type(value)==torch.Tensor:
            value = value.item()

        if not hasattr(self, title):
            setattr(self, title, self.vis.line([value], [0], opts=dict(title=title)))
            setattr(self, title+'_step', 1)
        else:
            step = getattr(self, title+'_step')
            self.vis.line([value], [step], win=getattr(self,title), update='append')
            setattr(self, title+'_step',step+1)

    def update_metrics(self, record):

        for title, value in record.items():
            self.update_line(title, value._metric)

    def new_text_window(self, title):
        win = self.vis.text(title)

        return win

    def append_text(self, text, win):
        self.vis.text(text, win=win, append=True)


class EarlyStopManager(object):
    def __init__(self, flags_obj):
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, metric, epoch):
        if epoch <=10:
            return False

        if metric > self.max_metric:
            self.max_metric = metric
            self.count = 0
            return False
        else:
            self.count += 1
            if self.count < self.es_patience:
                return False
            else:
                return True


class ModelSelector(object):
    def __init__(self):
        pass

    @staticmethod
    def getModel(flags_obj, dataset, model_name, device):
        if model_name=='MF':
            return MF(flags_obj, dataset, device)
        elif model_name=='MBGCN':
            return MBGCN(flags_obj, dataset, device)
        else:
            raise ValueError('Model name is not correct!')
