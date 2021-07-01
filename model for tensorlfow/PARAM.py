dtt = 20; # time step in ms
Para1 = [round(1000/dtt)] # targetStart
numPara1 = len(Para1)
numRep = 20 # number of simulations

for paraInd1 in range(numPara1):
    para1 = Para1[paraInd1]
    for repInd in range(numRep):
        runfile('main.py', args='--gpu 0 --gpu_frac 0.0 --n_trials 500000 --mode "train"\
                --N 200 --P_inh 0.20  --apply_dale True --gain 1 --task "2stim"\
                --act "softplus" --loss_fn "l2" --tau 100.0 --dt 20 --output_dir ".\"')
        tf.reset_default_graph()

