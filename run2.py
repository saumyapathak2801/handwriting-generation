#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf

import argparse
import time
import os


# from LSTM import Model
# from LSTM import load_pretrained_model
import LSTM
from LSTM import *
from importlib import reload
reload(LSTM)
from dataloader import DataProcess
from sample import *


# In[ ]:


def init_args():
    
        args = {}
        args['rnn_size'] = 100 
        args['tsteps'] = 150 
        args['batch_size'] = 2 
        args['num_mixtures'] = 8 
        args['alphabet'] = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        args['tsteps_per_ascii'] = 25
        args['epochs'] = 100
        args['path'] = 'saved/model.ckpt'
        args['bias'] = 1.0
        args['logs_dir'] = './logs/'
        return args

def load_pretrained_model(model, path):
        global_step = 0
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            model.saver.restore(model.sess, load_path)
            #load_was_success = True
        except:
            load_was_success = False
        else:
            model.saver = tf.train.Saver(tf.global_variables())
            global_step = int(load_path.split('-')[-1])
            load_was_success = True
        return load_was_success, global_step
    
def train_model():
    sess = tf.Session()
    data_loader = DataProcess()
    args = init_args()
    model = Model(args)
    load_was_success, global_step = load_pretrained_model(model, args['save_path'])
    v_x, v_y, v_s, v_c = data_loader.get_validation_data()
    valid_inputs = {model.input_data: v_x, model.target_data: v_y, model.char_seq: v_c}
#     for epoch in range(args['epochs']):
#         if (epoch%5 == 0):
#             print("Running epoch:", epoch)
        
#         c0, c1, c2 = model.istate_cell0.c.eval(session=sess), model.istate_cell1.c.eval(session=sess), model.istate_cell2.c.eval(session=sess)
#         h0, h1, h2 = model.istate_cell0.h.eval(session=sess), model.istate_cell1.h.eval(session=sess), model.istate_cell2.h.eval(session=sess)
#         for bindex in range(args['batch_size']):
#             i = epoch*args['batch_size'] + bindex
#             start = time.time()
#             x, y = data_loader.get_next_batch()
#             model_input = [tf.squeeze(input_, [1]) for input_ in tf.split(model.input, self.tsteps, 1)]
#             out0 = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, model.istate_cell0, model.cell0, loop_function=None, scope='cell0')
    
    for e in range(global_step/args['batch_size'], args['epochs']):

        c0, c1, c2 = model.istate_cell0.c.eval(), model.istate_cell1.c.eval(), model.istate_cell2.c.eval()
        h0, h1, h2 = model.istate_cell0.h.eval(), model.istate_cell1.h.eval(), model.istate_cell2.h.eval()

        for b in range(global_step%args['batch_size'], args['batch_size']):
            print(b)

            i = e * args.nbatches + b
            if global_step is not 0 : i+=1 ; global_step = 0

            if i % args.save_every == 0 and (i > 0):
                model.saver.save(model.sess, args.save_path, global_step = i) ; logger.write('SAVED MODEL')

            start = time.time()
            x, y, s, c = data_loader.get_next_batch()

            feed = {model.input_data: x, model.target_data: y, model.char_seq: c,                     model.istate_cell0.c: c0, model.istate_cell1.c: c1, model.istate_cell2.c: c2,                     model.istate_cell0.h: h0, model.istate_cell1.h: h1, model.istate_cell2.h: h2}

            [train_loss, _] = model.sess.run([model.cost, model.train_op], feed)
            feed.update(valid_inputs)
            [valid_loss] = model.sess.run([model.cost], feed)
            
            running_average = running_average*remember_rate + train_loss*(1-remember_rate)

            end = time.time()
            if i % 10 is 0: logger.write("{}/{}, loss = {:.3f}, regloss = {:.5f}, valid_loss = {:.3f}, time = {:.3f}"                 .format(i, args.nepochs * args.nbatches, train_loss, running_average, valid_loss, end - start) )

def sample_model(args, logger=None):
    if args.text == '':
        strings = ['call me ishmael some years ago', 'A project by Sam Greydanus', 'mmm mmm mmm mmm mmm mmm mmm',             'What I cannot create I do not understand', 'You know nothing Jon Snow'] # test strings
    else:
        strings = [args.text]

    model = Model(args)

    load_was_success, global_step = load_pretrained_model(model, args['save_path'])

    if load_was_success:
        for s in strings:
            strokes = sample(s, model, args)

            w_save_path = '{}figures/iter-{}-w-{}'.format(args['logs_dir'], global_step, s[:10].replace(' ', '_'))
            g_save_path = '{}figures/iter-{}-g-{}'.format(args['logs_dir'], global_step, s[:10].replace(' ', '_'))
            l_save_path = '{}figures/iter-{}-l-{}'.format(args['logs_dir'], global_step, s[:10].replace(' ', '_'))

            gauss_plot(strokes, 'Heatmap for "{}"'.format(s), figsize = (2*len(s),4), save_path=g_save_path)
            line_plot(strokes, 'Line plot for "{}"'.format(s), figsize = (len(s),2), save_path=l_save_path)

    else:
        print("load failed, sampling canceled")

    if True:
        tf.reset_default_graph()
        time.sleep(args.sleep_time)
        sample_model(args, logger=logger)

            
            
            
            
            
            
            
            
            
    


# In[2]:


train_model()

