'''
    author: Peijie Sun 
    date: 04/18/2019
    src path: util/pytorch/type1_train.py
'''

import os, sys, shutil

from time import time
import numpy as np
import torch

from Logging import Logging

def start(conf, data, model, evaluate, record_id, pre_model='', bias_epoch=0, first_terminal_flag=1):

    # start gpu environment for pytorch
    path = "%s/%s_%s" % (conf.root_dir, conf.model_name, record_id)
    
    # define log name 
    log_path = '%s/_%s_%s.log' % (path, record_id, conf.model_name)
    # define tmp train model
    prefix_train_model = '%s/tmp_model/%s' % (path, conf.model_name)
    # create tmp_model store tmp models
    train_model_dir = '%s/tmp_model' % path

    record_str = ''
    record_str += 'record_id: %d\n' % record_id
    record_str += 'log_path: %s\n' % log_path
    record_str += 'data_name: %s\n' % conf.data_name
    record_str += 'model_name: %s\n' % conf.model_name

    # start to prepare data for training and evaluating
    ###============================== TASK FLAG ==============================###
    #data.initializeTask3Handle()
    exec('data.initialize%sHandle()' % conf.task)
    d_train, d_val, d_test = data.train, data.val, data.test

    print('System start to load data...')
    t0 = time()
    #d_train.task3InitializeData()
    exec('d_train.%sInitializeData(model)' % conf.task.lower())
    #d_val.task3InitializeData()
    exec('d_val.%sInitializeData(model)' % conf.task.lower())
    #d_test.task3InitializeData()
    exec('d_test.%sInitializeData(model)' % conf.task.lower())
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))
    ###============================== TASK FLAG ==============================###

    # set debug_flag=0, doesn't print any results
    log = Logging(log_path, debug_flag=1)
    log.record('%s Record Parameters %s\n' % ('='*20, '='*20))
    log.record(record_str)
    log.record('%s Record Parameters %s\n' % ('='*20, '='*20))
    #log.record('model:%s, record_id:%s, config:%s' % (conf.model_name, record_id, config_path))

    model.initOptimizer(model)
    
    # standard tensorflow running environment initialize
    if pre_model != '':
        log.record('%s Go On Traing %s' % ('='*20, '='*20))
        model.load_state_dict(torch.load(pre_model))
    elif conf.pretrain_flag == 1:
        log.record('%s Go On Traing %s' % ('*'*20, '*'*20))
        model.load_state_dict(torch.load(conf.pre_model))
    
    val_loss_dict = {}
    min_train_loss, min_val_loss, min_test_loss = 10, 10, 10
    min_train_record, min_val_record, min_test_record = [], [], []

    # Start Training !!!
    for tmp_epoch in range(1, conf.epochs+1):
        epoch = tmp_epoch + bias_epoch

        # optimize model with training data and compute train loss
        tmp_train_loss, tmp_val_loss, tmp_test_loss = [], [], []
        t0 = time()
        
        # Following is the first training epoch, no optimization
        while d_train.terminal_flag and epoch == 1:
            #d_train.task3GetBatch()
            exec('d_train.%sGetBatch()' % conf.task.lower())

            train_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                train_feed_dict[key] = d_train.data_dict[value]
            
            model.train(train_feed_dict)
            sub_train_loss = model.map_dict['out']['output_loss']
            
            tmp_train_loss.append(sub_train_loss)
        train_loss = np.mean(tmp_train_loss)
        t1 = time()
        d_train.terminal_flag = 1

        # Following is the training process
        while d_train.terminal_flag and epoch > 1:
            #import cProfile
            #cProfile.runctx('d_train.task3GetBatch()', globals(), locals())
            #d_train.task3GetBatch()
            exec('d_train.%sGetBatch()' % conf.task.lower())

            train_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                train_feed_dict[key] = d_train.data_dict[value]
            
            model.train(train_feed_dict)
            model.optim()
            sub_train_loss = model.map_dict['out']['output_loss']
            
            tmp_train_loss.append(sub_train_loss)
        train_loss = np.mean(tmp_train_loss)
        t1 = time()
        d_train.terminal_flag = 1

        # Following is used to compute the loss of val and test dataset
        while d_val.terminal_flag:
            #d_val.task3GetBatch()
            exec('d_val.%sGetBatch()' % conf.task.lower())

            val_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                val_feed_dict[key] = d_val.data_dict[value]
            
            model.train(val_feed_dict)
            sub_val_loss = model.map_dict['out']['output_loss']
            torch.cuda.empty_cache()

            tmp_val_loss.append(sub_val_loss)
        val_loss = np.mean(tmp_val_loss)
        val_loss_dict[tmp_epoch] = val_loss
        #print(np.sum(tmp_val_loss))
        d_val.terminal_flag = 1

        while d_test.terminal_flag:
            #d_test.task3GetBatch()
            exec('d_test.%sGetBatch()' % conf.task.lower())

            test_feed_dict = {}
            for (key, value) in model.map_dict['input'].items():
                test_feed_dict[key] = d_test.data_dict[value]
            
            model.train(test_feed_dict)
            sub_test_loss = model.map_dict['out']['output_loss']
            torch.cuda.empty_cache()

            tmp_test_loss.append(sub_test_loss)
        test_loss = np.mean(tmp_test_loss)
        #print(np.sum(tmp_test_loss))
        d_test.terminal_flag = 1
        t2 = time()

        # record min train/val/test loss
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_record = [train_loss, val_loss, test_loss, epoch]
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_record = [train_loss, val_loss, test_loss, epoch]
        if test_loss < min_test_loss:
            min_test_loss =  test_loss
            min_test_record = [train_loss, val_loss, test_loss, epoch]

        if epoch % 50 == 0:
            torch.save(model.state_dict(), \
                "%s_loss_#%.4f#_epoch_#%d" % (prefix_train_model, train_loss, epoch))

        if tmp_epoch > 10 and \
            val_loss_dict[tmp_epoch] >= val_loss_dict[tmp_epoch-1] and \
            val_loss_dict[tmp_epoch-1] >= val_loss_dict[tmp_epoch-2]:
            #torch.save(model.state_dict(), \
            #    "%s_loss_#%.4f#_epoch_#%d" % (prefix_train_model, train_loss, epoch+bias_epoch))
            #clean.cleanRatingModelCache(train_model_dir, min_test_loss)
            log.record('%s%s%s' % ('-'*35, '*'*20, '-'*35))
            log.record('%s%sEarly Stop%s%s' % ('-'*35, '*'*5, '*'*5, '-'*35))
            log.record('%s%s%s' % ('-'*35, '*'*20, '-'*35))
            #earlystop_flag = 0
                
        # print log to console and log_file
        log.record('Current Model is:%s, Record ID is:%s' % (conf.model_name, record_id))
        log.record('Epoch:%s' % (epoch))
        log.record('NLL: train:%.4f, val:%.4f, test:%.4f' % (train_loss, val_loss, test_loss))
        log.record('Time: %.4fs, train cost:%.4fs, validation cost:%.4fs' % ((t2-t0), (t1-t0), (t2-t1)))

    log.record('%s Results Summarization %s' % ('='*20, '='*20))
    log.record('Current Model is:%s, Record ID is:%s' % (conf.model_name, record_id))
    log.record('Minimum train loss, epoch:%d, train loss:%.4f, val loss:%.4f, test loss:%.4f' % \
        (min_train_record[3], min_train_record[0], min_train_record[1], min_train_record[2]))
    log.record('Minimum val loss, epoch:%d, train loss:%.4f, val loss:%.4f, test loss:%.4f' % \
        (min_val_record[3], min_val_record[0], min_val_record[1], min_val_record[2]))
    log.record('Minimum test loss, epoch:%d, train loss:%.4f, val loss:%.4f, test loss:%.4f' % \
        (min_test_record[3], min_test_record[0], min_test_record[1], min_test_record[2]))
    log.record('%s Results Summarization %s' % ('='*20, '='*20))