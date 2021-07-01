# Training firing rate RNN to do two two-interval timg task:2-Stimuli(2stim) and 2-Context(2con)
# started from codes from: Robert Kim et al., Simple framework for constructing functional spiking recurrent neural networks, PNAS, 2019
# by Shanglin Zhou and Dean Buonomano

import os, sys
import time
import numpy as np
import tensorflow as tf
import argparse
import datetime
import scipy.io

# Import utility functions
from utils import set_gpu
from utils import restricted_float
from utils import str2bool

# Import the continuous rate model
from model import FR_RNN_dale

# Import the tasks
from model import generate_input_stim_2stim
from model import generate_target_continuous_2stim

from model import generate_input_stim_2con
from model import generate_target_continuous_2con

from model import construct_tf
from model import loss_op

# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N", required=True,
        type=int, help="Number of neurons")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--task", required=True,
        type=str, help="Task (2stim, 2con, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--tau", required=True,
        nargs='+', type=float,
        help="time constant ")
parser.add_argument("--dt", required=True,
        type=int, default = 10,
        help="time step")
args = parser.parse_args()

# Set up the output dir where the results will be saved to 
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

# Number of units
N = args.N


# Define task-specific parameters
if args.task.lower() == '2stim':
     # 2-Stimuli task
    settings = {
            'T': round((600 + 500 + 5500 + 200)/args.dt), # trial duration for long interval (in steps)
            'Ts': round((600 + 500 + 2500 + 200)/args.dt), # trial duration for short interval (in steps)
            'stim_on': round(600/args.dt), # time when the simulus starts
            'stim_dur': round(500/args.dt), # duration of the stimulus
            'interval1': round(2500/args.dt), # length of the short interval
            'interval2': round(5500/args.dt), # length of the long interval
            'extended_dur':round(200/args.dt),# duration of the Plateau for the target
            'DeltaT': args.dt, # time step
            'tau': args.tau, # time constant of unit
            'task': args.task.lower(), # task name
            'targetStart':para1, # time when the target ramp starts for the shrot interval
            }
    
elif args.task.lower() == '2con':
    # 2-Context task
    settings = {
            'T': round((600 + 500 + 5500 + 200)/args.dt), # trial duration for long interval (in steps)
            'Ts': round((600 + 500 + 2500 + 200)/args.dt), # trial duration for short interval (in steps)
            'stim_on': round(600/args.dt), # time when the simulus starts
            'stim_dur': round(500/args.dt), #  # duration of the go stimulus
            'interval1': round(2500/args.dt), # length of the short interval 
            'interval2': round(5500/args.dt), # # length of the long interval
            'extended_dur':round(200/args.dt),# duration of the Plateau for the target
            'DeltaT': args.dt, # time step
            'tau': args.tau, # time constant of unit
            'task': args.task.lower(), # task name
            'targetStart':para1, # time when the target ramp starts for the shrot interval
            }


'''
Initialize the input and output weight matrices
'''  
if args.task.lower() == '2stim':
    w_in = np.float32(np.random.randn(N, 2))
    w_out = np.float32(np.random.randn(1, N)/100)
    
elif args.task.lower() == '2con':
    w_in = np.float32(np.random.randn(N, 2))
    w_out = np.float32(np.random.randn(1, N)/100)

'''
Initialize the model
'''
P_inh = args.P_inh # inhibitory neuron proportion
P_rec = args.P_rec # initial connectivity probability
print('P_rec set to ' + str(P_rec))

w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
net = FR_RNN_dale(N, P_inh, P_rec, w_in, w_dist, args.gain, args.apply_dale, w_out)
print('Intialized the network...')


'''
Define the training parameters (learning rate, training termination criteria, etc...)
'''
training_params = {
        'learning_rate': 0.01, # learning rate 0.01
        'loss_threshold': 2, # loss threshold (when to stop training)
        'eval_freq': 100, # how often to evaluate task performance
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.6, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': 0.20,# initial connectivity probability
        }


'''
Construct the TF graph for training
'''
if args.mode.lower() == 'train':
    input_node, z, x, r, o, w, w_in, m, w_out, b_out\
            = construct_tf(net, settings, training_params)
    print('Constructed the TF graph...')

    # Loss function and optimizer
    loss, training_op = loss_op(r, o, z, training_params,settings)
    
'''
Start the TF session and train the network
'''
sess = tf.Session(config=tf.ConfigProto(gpu_options=set_gpu(args.gpu, args.gpu_frac)))
init = tf.global_variables_initializer()

if args.mode.lower() == 'train':
    with tf.Session() as sess:
        print('Training started...')
        init.run()
        training_success = False

        if args.task.lower() == '2stim':
            settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
            u, label = generate_input_stim_2stim(settings)
            target = generate_target_continuous_2stim(settings, label)
            x0, r0, w0, w_in0, w_out0 = \
                    sess.run([x, r, w, w_in, w_out], feed_dict={input_node: u, z: target})
                    
        elif args.task.lower() == '2con':
            settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
            u, label = generate_input_stim_2con(settings)
            target = generate_target_continuous_2con(settings, label)
            x0, r0, w0, w_in0,w_out0 = \
                    sess.run([x, r, w, w_in, w_out], feed_dict={input_node: u, z: target})
                    
        # For storing all the loss vals
        losses = np.zeros((args.n_trials,))
        perfEvals = np.zeros((args.n_trials,));
        lossEvals = np.zeros((args.n_trials,));
        evalCount = 0;
        
        for tr in range(args.n_trials):
            start_time = time.time()

            # Generate a task-specific input signal
            if args.task.lower() == '2stim':
                settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
                u, label = generate_input_stim_2stim(settings)
                target = generate_target_continuous_2stim(settings, label)
            elif args.task.lower() == '2con':
                settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
                u, label = generate_input_stim_2con(settings)
                target = generate_target_continuous_2con(settings, label)


            # Train using backprop
            _, t_loss, t_w, t_o, t_w_out, t_x, t_r, t_m, t_w_in, t_b_out = \
                    sess.run([training_op, loss, w, o, w_out, x, r, m, w_in, b_out],
                    feed_dict={input_node: u, z: target})

            losses[tr] = t_loss

            '''
            Evaluate the model and determine if the training termination criteria are met
            '''

            if args.task.lower() == '2stim':
                if (tr)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']))
                    eval_labels = []
                    eval_us = np.zeros((training_params['eval_tr'],2,settings['T']))
                    eval_rs = np.zeros((training_params['eval_tr'],N,settings['T']))
                    eval_zs = np.zeros((training_params['eval_tr'], settings['T']))
                    
                    for ii in range(eval_perf.shape[-1]):
                        settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
                        eval_u, eval_label = generate_input_stim_2stim(settings)
                        eval_target = generate_target_continuous_2stim(settings, eval_label)
                        eval_o, eval_r, eval_l = sess.run([o, r, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels.append(eval_label)
                        
                        eval_us[ii,:,:] = eval_u
                        eval_rs[ii,:,:] = np.transpose(np.squeeze(np.array(eval_r)))
                        eval_zs[ii, :] = eval_target
                        
                        T = settings['T']
                        Ts = settings['Ts']
                        stim_on = settings['stim_on']
                        stim_dur = settings['stim_dur']
                        targetStart = settings['targetStart']
                        interval1 = settings['interval1']
                        interval2 = settings['interval2']
                        T1 = stim_on + stim_dur + interval1
                        T2 = stim_on + stim_dur + interval2
                        
                        temp = np.array(eval_o)
                        if eval_label == 'short':
                            resp = np.array(np.where(temp[stim_on:T1]> training_params['eval_amp_threh']))
                            if resp.size > 0:
                                if resp[0][0]>(stim_dur + targetStart):
                                    eval_perf[0, ii] = 1
                        else:
                            resp = np.array(np.where(temp[stim_on:T2]> training_params['eval_amp_threh']))
                            if resp.size > 0:
                                if resp[0][0]>(stim_dur*2 + targetStart*2):
                                    eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    perfEvals[evalCount] = eval_perf_mean
                    lossEvals[evalCount] = eval_loss_mean
                    evalCount = evalCount + 1
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))
                    
                    

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean>0.97:
                        training_success = True
                        break
                                      
            elif args.task.lower() == '2con':
                if (tr)%training_params['eval_freq'] == 0:
                    eval_perf = np.zeros((1, training_params['eval_tr']))
                    eval_losses = np.zeros((1, training_params['eval_tr']))
                    eval_os = np.zeros((training_params['eval_tr'], settings['T']))
                    eval_labels = []
                    eval_us = np.zeros((training_params['eval_tr'],2,settings['T']))
                    eval_rs = np.zeros((training_params['eval_tr'],N,settings['T']))
                    eval_zs = np.zeros((training_params['eval_tr'], settings['T']))
                    
                    for ii in range(eval_perf.shape[-1]):
                        settings['stim_on'] = np.int32(np.random.random_sample()*400/args.dt + 200/args.dt)
                        eval_u, eval_label = generate_input_stim_2con(settings)
                        eval_target = generate_target_continuous_2con(settings, eval_label)
                        eval_o, eval_r, eval_l = sess.run([o, r, loss], feed_dict = \
                                {input_node: eval_u, z: eval_target})
                        eval_losses[0, ii] = eval_l
                        eval_os[ii, :] = eval_o
                        eval_labels.append(eval_label)
                        
                        eval_us[ii,:,:] = eval_u
                        eval_rs[ii,:,:] = np.transpose(np.squeeze(np.array(eval_r)))
                        eval_zs[ii, :] = eval_target
                        
                        
                        T = settings['T']
                        Ts = settings['Ts']
                        stim_on = settings['stim_on']
                        stim_dur = settings['stim_dur']
                        targetStart = settings['targetStart']
                        interval1 = settings['interval1']
                        interval2 = settings['interval2']
                        T1 = stim_on + stim_dur + interval1
                        T2 = stim_on + stim_dur + interval2
                        
                        temp = np.array(eval_o)
                        if eval_label == 'short':
                            resp = np.array(np.where(temp[stim_on:T1]> training_params['eval_amp_threh']))
                            if resp.size > 0:
                                if resp[0][0]>(stim_dur + targetStart):
                                    eval_perf[0, ii] = 1
                        else:
                            resp = np.array(np.where(temp[stim_on:T2]> training_params['eval_amp_threh']))
                            if resp.size > 0:
                                if resp[0][0]>(stim_dur*2 + targetStart*2):
                                    eval_perf[0, ii] = 1

                    eval_perf_mean = np.nanmean(eval_perf, 1)
                    eval_loss_mean = np.nanmean(eval_losses, 1)
                    perfEvals[evalCount] = eval_perf_mean
                    lossEvals[evalCount] = eval_loss_mean
                    evalCount = evalCount + 1
                    print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean, eval_loss_mean))
                    
                    

                    if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean>0.97:
                        training_success = True
                        break 

        elapsed_time = time.time() - start_time
        # print(elapsed_time)

        # Save the trained params in a .mat file
        var = {}
        var['args'] = args
        var['x0'] = x0
        var['r0'] = r0
        var['w0'] = w0
        var['training_params'] = training_params
        var['settings'] = settings
        var['u'] = u
        var['o'] = t_o
        var['w'] = t_w
        var['x'] = t_x
        var['target'] = target
        var['w_out'] = t_w_out
        var['w_out0'] = w_out0
        var['r'] = t_r
        var['m'] = t_m
        var['N'] = N
        var['exc'] = net.exc
        var['inh'] = net.inh
        var['w_in'] = t_w_in
        var['w_in0'] = w_in0
        var['b_out'] = t_b_out
        var['losses'] = losses
        
        var['perfEvals'] = perfEvals
        var['lossEvals'] = lossEvals
        var['eval_os'] = eval_os
        var['eval_labels'] = eval_labels
        var['eval_us'] = eval_us
        var['eval_rs'] = eval_rs
        var['eval_zs'] = eval_zs
        
        var['tr'] = tr
        var['activation'] = training_params['activation']
        fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

        fname = 'Task_{}_Para1_{}_rep_{}.mat'.format(args.task.lower(), paraInd1, 
                    repInd)            
        scipy.io.savemat(os.path.join(out_dir, fname), var)


