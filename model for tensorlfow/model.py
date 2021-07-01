import os, sys
import numpy as np
import tensorflow as tf
import scipy.io

'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale:
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, N, P_inh, P_rec, w_in, w_dist, gain, apply_dale, w_out):
        """
        Network initialization method
        N: number of units (neurons)
        P_inh: probability of a neuron being inhibitory
        P_rec: recurrent connection probability
        w_in: NxN weight matrix for the input stimuli
        w_dist: recurrent weight distribution ('gaus' or 'gamma')
        apply_dale: apply Dale's principle ('True' or 'False')
        w_out: Nx1 readout weights

        Based on the probability (P_inh) provided above,
        the units in the network are classified into
        either excitatory or inhibitory. Next, the
        weight matrix is initialized based on the connectivity
        probability (P_rec) provided above.
        """
        self.N = N
        self.P_inh = P_inh
        self.P_rec = P_rec
        self.w_in = w_in
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale
        self.w_out = w_out

        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE = self.assign_exc_inh()
        self.inh = inh
        self.exc = exc
        self.NI = NI
        self.NE = NE

        # Initialize the weight matrix
        self.W, self.mask = self.initialize_W()

    def assign_exc_inh(self):
        """
        Method to randomly assign units as excitatory or inhibitory (Dale's principle)

        Returns
            inh: bool array marking which units are inhibitory
            exc: bool array marking which units are excitatory
            NI: number of inhibitory units
            NE: number of excitatory units
        """
        # Apply Dale's principle
        if self.apply_dale == True:
            inh = np.random.rand(self.N, 1) < self.P_inh
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI

        # Do NOT apply Dale's principle
        else:
            inh = np.random.rand(self.N, 1) < 0 # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI


        return inh, exc, NI, NE

    def initialize_W(self):
        """
        Method to generate and initialize the connectivity weight matrix, W
        The weights are drawn from either gaussian or gamma distribution.

        Returns
            w: NxN weights (all positive)
            mask: NxN matrix of 1's (excitatory units)
                  and -1's (for inhibitory units)
        NOTE: To compute the "full" weight matrix, simply
        multiply w and mask (i.e. w*mask)
        """
        # Weight matrix
        w = np.zeros((self.N, self.N), dtype = np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w[idx[0], idx[1]] = np.random.gamma(2, 0.003, len(idx[0]))
        elif self.w_dist.lower() == 'gaus':
            w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            w = w/np.sqrt(self.N*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w = np.abs(w)
        w[np.diag_indices(self.N,2)] = 0.
        # Mask matrix
        mask = np.eye((self.N), dtype=np.float32)
        mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1
        
        mask_ini = np.eye((self.N), dtype=np.float32)
        mask_ini[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = ((1-self.P_inh)/self.P_inh)
        w = np.matmul(w,mask_ini) # set E/I balance to w

        return w, mask

    def load_net(self, model_dir):
        """
        Method to load pre-configured network settings
        """
        settings = scipy.io.loadmat(model_dir)
        self.N = settings['N'][0][0]
        self.inh = settings['inh']
        self.exc = settings['exc']
        self.inh = self.inh == 1
        self.exc = self.exc == 1
        self.NI = len(np.where(settings['inh'] == True)[0])
        self.NE = len(np.where(settings['exc'] == True)[0])
        self.mask = settings['m']
        self.W = settings['w']
        self.w_in = settings['w_in']
        self.b_out = settings['b_out']
        self.w_out = settings['w_out']

        return self
    
    def display(self):
        """
        Method to print the network setup
        """
        print('Network Settings')
        print('====================================')
        print('Number of Units: ', self.N)
        print('\t Number of Excitatory Units: ', self.NE)
        print('\t Number of Inhibitory Units: ', self.NI)
        print('Weight Matrix, W')
        full_w = self.W*self.mask
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N*self.N)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N*self.N)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N*self.N)*100))

'''
Task-specific input signals
'''

def generate_input_stim_2stim(settings):
    """
    Method to generate the input stimulus matrix (u)
    for 2-Stimuli task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            tau: time-constants (in ms)
            DeltaT: sampling rate
    OUTPUT
        u: 2xT stimulus matrix
        label: 'short' or 'long'
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    interval1 = settings['interval1']
    interval2 = settings['interval2']
    T1 = stim_on + stim_dur + interval1
    T2 = stim_on + stim_dur + interval2

    # Initialize u
    u = np.zeros((2, T))

    inputGain = 1;
    if np.random.rand() < 0.5:
        u[0, stim_on:stim_on+stim_dur] = 1*inputGain
        label = 'short'
    else:
        u[1, stim_on:stim_on+stim_dur] = 1*inputGain
        label = 'long'

    return u, label

def generate_input_stim_2con(settings):
    """
    Method to generate the input stimulus matrix (u)
    for 2-Context task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            tau: time-constants (in ms)
            DeltaT: sampling rate
    OUTPUT
        u: 2xT stimulus matrix
        label: 'short' or 'long'
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    interval1 = settings['interval1']
    interval2 = settings['interval2']
    T1 = stim_on + stim_dur + interval1
    T2 = stim_on + stim_dur + interval2
    extended_dur = settings['extended_dur']


    contextShort = settings['contextShort']

    # Initialize u
    u = np.zeros((2, T))

    # XOR task

    inputGain = 1;
    if np.random.rand() < 0.5:
        u[0, stim_on:stim_on+stim_dur] = 1*inputGain
        u[1, stim_on:T1+extended_dur] = u[1, stim_on:T1+ extended_dur]  + contextShort;
        label = 'short'
#        label = 'long'
    else:
        u[0, stim_on:stim_on+stim_dur] = 1*inputGain
        u[1, stim_on:T2+ extended_dur] = u[1, stim_on:T2+extended_dur] + 1  - contextShort;
        label = 'long'
#        label = 'short'

    return u, label
'''
Task-specific target signals
'''

def generate_target_continuous_2stim(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the 2-Stimuli task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            delay: delay b/w two stimuli (in steps)
            tau: time-constants (in ms)
            DeltaT: sampling rate
        label: string value (either 'short' or 'long')
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    interval1 = settings['interval1']
    interval2 = settings['interval2']
    T1 = stim_on + stim_dur + interval1
    T2 = stim_on + stim_dur + interval2
    targetStart = settings['targetStart']
    extended_dur = settings['extended_dur']
    
    if label == 'long':
        z = np.zeros((1, T))
        z[0, stim_on + stim_dur*2 + targetStart*2:T2] = np.linspace(0,1,T2 - targetStart*2 - stim_on - stim_dur*2)
        z[0, T2:T2+extended_dur] = 1
    elif label == 'short':
        z = np.zeros((1, T))
        z[0,  stim_on + stim_dur +targetStart:T1] =  np.linspace(0,1,T1 - targetStart - stim_on - stim_dur)
        z[0, T1:T1+extended_dur] = 1

    return np.squeeze(z)

def generate_target_continuous_2con(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the 2-Context task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            tau: time-constants (in ms)
            DeltaT: sampling rate
        label: string value (either 'same' or 'diff')
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    interval1 = settings['interval1']
    interval2 = settings['interval2']
    T1 = stim_on + stim_dur + interval1
    T2 = stim_on + stim_dur + interval2
    targetStart = settings['targetStart']
    extended_dur = settings['extended_dur']

    
    if label == 'long':
        z = np.zeros((1, T))
        z[0, stim_on + stim_dur*2 + targetStart*2:T2] = np.linspace(0,1,T2 - targetStart*2 - stim_on - stim_dur*2)
        z[0, T2:T2+extended_dur] = 1
    elif label == 'short':
        z = np.zeros((1, T))
        z[0,  stim_on + stim_dur +targetStart:T1] =  np.linspace(0,1,T1 - targetStart - stim_on - stim_dur)
        z[0, T1:T1+extended_dur] = 1
    return np.squeeze(z)

'''
CONSTRUCT TF GRAPH FOR TRAINING
'''
def construct_tf(fr_rnn, settings, training_params):
    """
    Method to construct a TF graph and return nodes with
    Dale's principle
    INPUT
        fr_rnn: firing-rate RNN class
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            tau: time-constants (in ms)
            DeltaT: sampling rate
        training_params: dictionary containing training parameters
            learning_rate: learning rate
    OUTPUT
        TF graph
    """

    # Task params
    T = settings['T']
    tau = settings['tau'][0]
    DeltaT = settings['DeltaT']
    task = settings['task']

    # Training params
    learning_rate = training_params['learning_rate']

    # Excitatory units
    exc_idx_tf = tf.constant(np.where(fr_rnn.exc == True)[0], name='exc_idx')

    # Inhibitory units
    inh_idx_tf = tf.constant(np.where(fr_rnn.inh == True)[0], name='inh_idx')

    # Input node

      
    if task == '2stim':
        stim = tf.placeholder(tf.float32, [2, T], name='u')
        
    elif task == '2con':
        stim = tf.placeholder(tf.float32, [2, T], name='u')
        
    # Target node
    z = tf.placeholder(tf.float32, [T,], name='target')



    # Synaptic currents and firing-rates
    x = [] # synaptic currents
    r = [] # firing-rates
#    x.append(tf.random_normal([fr_rnn.N, 1], dtype=tf.float32)/10)
    x.append(tf.random_normal([fr_rnn.N, 1], dtype=tf.float32)/10)

    # Transfer function options
    if training_params['activation'] == 'sigmoid':
        r.append(tf.sigmoid(x[0]))
    elif training_params['activation'] == 'clipped_relu': 
        r.append(tf.clip_by_value(tf.nn.relu(x[0]), 0, 20))
    elif training_params['activation'] == 'softplus':
        r.append(tf.clip_by_value(tf.nn.softplus(x[0]), 0, 20))

    # Initialize recurrent weight matrix, mask, input & output weight matrices
    w = tf.get_variable('w', initializer = fr_rnn.W, dtype=tf.float32, trainable=True)
    m = tf.get_variable('m', initializer = fr_rnn.mask, dtype=tf.float32, trainable=False)
    w_in = tf.get_variable('w_in', initializer = fr_rnn.w_in, dtype=tf.float32, trainable=False)
    w_out = tf.get_variable('w_out', initializer = fr_rnn.w_out, dtype=tf.float32, 
            trainable=True)

    b_out = tf.Variable(0, dtype=tf.float32, name='b_out', trainable=False)

    # Forward pass
    o = [] # output (i.e. weighted linear sum of rates, r)
    o.append(tf.matmul(w_out, r[0]) + b_out)
    
    for t in range(1, T):
        if fr_rnn.apply_dale == True:
            # Parametrize the weight matrix to enforce exc/inh synaptic currents
            w = tf.nn.relu(w)

        # next_x is [N x 1]
        ww = tf.matmul(w, m)

        # Pass the synaptic time constants thru the sigmoid function

        next_x = tf.multiply((1 - DeltaT/tau), x[t-1]) + \
                tf.multiply((DeltaT/tau), ((tf.matmul(ww, r[t-1]))\
                + tf.matmul(w_in, tf.expand_dims(stim[:, t], 1))) +\
                tf.random_normal([fr_rnn.N, 1], dtype=tf.float32)*tf.math.sqrt(2*tau)*0.1) 
               
        x.append(next_x)

        if training_params['activation'] == 'sigmoid':
            r.append(tf.sigmoid(next_x))
        elif training_params['activation'] == 'clipped_relu': 
            r.append(tf.clip_by_value(tf.nn.relu(next_x), 0, 20))
        elif training_params['activation'] == 'softplus':
            r.append(tf.clip_by_value(tf.nn.softplus(next_x), 0, 20))

        next_o = tf.matmul(w_out, r[t]) + b_out
        o.append(next_o)

    return stim, z, x, r, o, w, w_in, m, w_out, b_out

'''
DEFINE LOSS AND OPTIMIZER
'''
def loss_op(r, o, z, training_params, settings):
    """
    Method to define loss and optimizer for ONLY ONE target signal
    INPUT
        o: list of output values
        z: target values
        training_params: dictionary containing training parameters
            learning_rate: learning rate

    OUTPUT
        loss: loss function
        training_op: optimizer
    """
    # Loss function
    loss = tf.zeros(1)
    loss_fn = training_params['loss_fn']
    mask_short = np.zeros(settings['T'])
    mask_short[:(settings['stim_on'] + settings['stim_dur'] + settings['interval1']+settings['extended_dur'])  ] = 1;#Ts; dosn't care after short interval
    mask_long = np.zeros(settings['T'])
    mask_long[:(settings['stim_on'] + settings['stim_dur'] + settings['interval2']+settings['extended_dur']) ] = 1;#Ts; dosn't care after short interval
    for i in range(0, len(o)):
        if loss_fn.lower() == 'l1':
            loss += tf.norm(o[i] - z[i])
        elif loss_fn.lower() == 'l2':
            if z[-70] == 0:
               loss += tf.multiply(tf.square(o[i] - z[i]),mask_short[i])
#               loss += tf.square(o[i] - z[i])
            else:
#               loss += tf.multiply(tf.square(o[i] - z[i]),mask_same[i])
               loss += tf.multiply(tf.square(o[i] - z[i]),mask_long[i])
    if loss_fn.lower() == 'l2':
        loss = tf.sqrt(loss)
        

    # Optimizer function
    with tf.name_scope('ADAM'):
        optimizer = tf.train.AdamOptimizer(learning_rate = training_params['learning_rate']) 
    
        training_op = optimizer.minimize(loss) 
    
    return loss, training_op
