#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME') 
DATA_DIR = '/sdf/home/d/dcarber/DATADIR'
# Set software directory
sys.path.append(SOFTWARE_DIR)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')


import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)


# In[3]:


from mlreco.main_funcs import process_config, prepare
import warnings, yaml
warnings.filterwarnings('ignore')

cfg = yaml.load(open('%s/inference_full_chain_020223.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
#'%s/inference_full_volume_112022.cfg'
#cfg = yaml.load(open('%s/inference_full_volume_112022.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
process_config(cfg, verbose=False)


# In[4]:


# prepare function configures necessary "handlers"
hs = prepare(cfg)
dataset = hs.data_io_iter


# In[5]:


data, result = hs.trainer.forward(dataset)


# In[6]:


print(data.keys())
print(data["index"])


# In[8]:


from analysis.classes.ui import FullChainEvaluator


# In[9]:


# Only run this cell once!
evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
#evaluator = FullChainEvaluator(data, result, full_chain_cfg, predictor_cfg, **kwarg)         

print(evaluator)


# In[ ]:


entry = 4# Batch ID for current sample
print("Batch ID = ", evaluator.index[entry])
print("Index", evaluator.index)


# In[ ]:


particles = evaluator.get_particles(entry, only_primaries=True)
true_particles = evaluator.get_true_particles(entry, only_primaries=False)
#print(evaluator.get_true_interactions(entry,volume=0))
matched_particles = evaluator.match_particles(entry, only_primaries = True,mode = 'pred_to_true')
from pprint import pprint
for t in true_particles:
    print(t.particle_asis.pdg_code())
    print(t.interaction_id)
pprint(particles)
pprint(true_particles)
#pprint(particles[0].interaction_id)
#pprint(matched_particles)


# In[ ]:


def column(matrix, i):
    return [row[i] for row in matrix]


from mlreco.visualization.plotly_layouts import white_layout, trace_particles, trace_interactions



import pandas as pd
def particles_counter(true_particles,particles):
    p_num_e = 0
    t_num_e = 0
    p_num_p = 0
    t_num_p = 0
    p_num_m = 0
    t_num_m = 0
    p_num_pi = 0
    t_num_pi = 0
    p_num_pr = 0
    t_num_pr = 0
    for tp in true_particles:
        if tp.pid == 0:
            t_num_p +=1
        if tp.pid == 1:
            t_num_e +=1
        if tp.pid == 2:
            t_num_m +=1
        if tp.pid == 3:
            t_num_pi +=1
        if tp.pid == 4:
            t_num_pr +=1
    for rp in particles:
        if rp.pid == 0:
            p_num_p +=1
        if rp.pid == 1:
            p_num_e +=1
        if rp.pid == 2:
            p_num_m +=1
        if rp.pid == 3:
            p_num_pi +=1
        if rp.pid == 4:
            p_num_pr +=1

    return p_num_e,p_num_p,p_num_m,p_num_pi,p_num_pr,t_num_e,t_num_p,t_num_m,t_num_pi,t_num_pr
def inter_type_loader(matched_particles,interact,true_primaries):
    p_pid = []
    p_type = []
    t_pid = []
    t_type = []
    size_diff = []
    particles = []
    true_particles = []
    int_reco = []
    int_true = []
    true_int =[]
    reco_int = []
    r_zero=0
    r_one = 0
    r_two = 0
    r_three = 0
    r_four = 0
    r_five = 0 
    r_six = 0
    r_seven = 0

    t_zero=0
    t_one = 0
    t_two = 0
    t_three = 0
    t_four = 0
    t_five = 0
    t_six = 0
    t_seven = 0
    for m in matched_particles: #Seperates particles and their matched true particle
        particles.append(m[0])
        true_particles.append(m[1])
    for m_int in interact:
        true_int.append(m_int[0].id)
        if m_int[1] == None:
            reco_int.append(None)
            continue
        reco_int.append(m_int[1].id)
    for i in range(len(true_int)):
        t_e = 0
        t_p = 0
        t_y = 0
        t_m = 0
        t_pi = 0
        r_e = 0
        r_p = 0
        r_y = 0
        r_m = 0
        r_pi = 0
        t_none = 0
        r_none = 0
        
        for r in range(len(matched_particles)):
            if true_particles[r] == None:
                print("No true")
                if particles[r].interaction_id ==reco_int[i]:
                    if particles[r].pid == 0:
                        print("ry")
                        r_y +=1
                        t_none +=1
                    if particles[r].pid == 1:
                        print("re")
                        r_e +=1
                        t_none +=1
                    if particles[r].pid == 2:
                        print("rm")
                        r_m +=1
                        t_none +=1
                    if particles[r].pid == 3:
                        print("rpi")
                        r_pi +=1
                        t_none +=1
                    if particles[r].pid == 4:
                        print("rp")
                        r_p+=1
                        t_none +=1

                continue
            #print("interaction",true_int[i], true_particles[r].interaction_id,particles[r].interaction_id)
            if true_particles[r].interaction_id ==true_int[i]:
                if true_particles[r].pid == 0:
                    print("ty")
                    t_y +=1
                if true_particles[r].pid == 1:
                    print("te")
                    t_e +=1
                if true_particles[r].pid == 2:
                    print("tm")
                    t_m +=1
                if true_particles[r].pid == 3:
                    print("tpi")
                    t_pi +=1
                if true_particles[r].pid == 4:
                    print("tp")
                    t_p+=1
            if particles[r].interaction_id ==reco_int[i]:
                if particles[r].pid == 0:
                    print("ry")
                    r_y +=1
                if particles[r].pid == 1:
                    print("re")
                    r_e +=1
                if particles[r].pid == 2:
                    print("rm")
                    r_m +=1
                if particles[r].pid == 3:
                    print("rpi")
                    r_pi +=1
                if particles[r].pid == 4:
                    print("rp")
                    r_p+=1

        '''           
        if t_e == 1 and t_y ==0:
            int_true.append(0)
            print(0)
        elif t_e == 0 and t_y ==1:
            int_true.append(1)
            print(1)
        elif t_e == 0 and t_y ==2:
            int_true.append(2)
            print(2)
        elif t_e == 2 and t_y ==0:
            int_true.append(3)
            print(3)
        elif t_e == 0 and t_y ==3:
            int_true.append(4)
            print(4)
        else:
            int_true.append(5)
            print(5)

        if r_e == 1 and r_y ==0:
            int_reco.append(0)
            print(0)
        elif r_e == 0 and r_y ==1:
            int_reco.append(1)
            print(1)
        elif r_e == 0 and r_y ==2:
            int_reco.append(2)
            print(2)
        elif r_e == 2 and r_y ==0:
            int_reco.append(3)
            print(3)
        elif r_e == 0 and r_y ==3:
            int_reco.append(4)
            print(4)
        else:
            int_reco.append(5)
            print(5)
    return int_reco,int_true
    '''
        
        if t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p == 0:
            int_true.append(0)
            #print(0)
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p ==1:
            int_true.append(1)
            #print(1)
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p >1 :
            int_true.append(2)
            #print(2)
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 1 and t_p == 0:
            int_true.append(3)
            #print(3)
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 1 and t_p == 1:
            int_true.append(4)
            #print(4)
        elif t_e == 0 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p == 1:
            int_true.append(5)
            #print(5)
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p == 0:
            int_true.append(6)
            print(6)
            if r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p == 0:
                print("Wrong e")
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p == 1:
            int_true.append(7)
            #print(7)
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p >1:
            int_true.append(8)
            #print(8)
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 1 and t_p ==0:
            int_true.append(9)
            #print(9)
        else:
            int_true.append(10)
            #print(10)

        if r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p == 0:
            int_reco.append(0)
            #print('t0')
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p == 1:
            int_reco.append(1)
            #print('t1')
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p >1:
            int_reco.append(2)
            #print('t2')
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 1 and r_p == 0:
            int_reco.append(3)
            #print('t3')
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 1 and r_p == 1:
            int_reco.append(4)
            #print('t4')
        elif r_e == 0 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p == 1:
            int_reco.append(5)
            #print('t5')
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p == 0:
            int_reco.append(6)
            #print('t6')
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p == 1:
            int_reco.append(7)
            #print('t7')
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p > 1:
            int_reco.append(8)
            #print('t8')
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 1 and r_p == 0:
            int_reco.append(9)
            #print('t9')
        else:
            int_reco.append(10)
            #print('t10')
    return int_reco,int_true
    
    '''
        if t_e == 1 and t_y ==0 and t_m == 0 and t_pi > 0 and t_pi < 4 and t_p > 1:
            int_true.append(0)
            print('t0')
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi ==0 and t_p > 1:
            int_true.append(1)
            print('t1')
        elif t_e == 1 and t_y == 0 and t_m == 0 and t_pi == 0 and t_p ==1 :
            int_true.append(2)
            print('t2')
        elif t_e == 1 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p == 0:
            int_true.append(3)
            print('t3')
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi >0 and t_p >1:
            int_true.append(4)
            print('t4')
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p >1:
            int_true.append(5)
            print('t5')
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p ==1:
            int_true.append(6)
            print('t6')
        elif t_e == 0 and t_y ==1 and t_m == 0 and t_pi == 0 and t_p ==0:
            int_true.append(7)
            print('t7')
        else:
            int_true.append(8)
            print('t8')
            
        if r_e == 1 and r_y ==0 and r_m == 0 and r_pi > 0 and r_p > 1 and r_p < 4:
            int_reco.append(0)
            print(0)
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p > 1 and r_p < 4:
            int_reco.append(1)
            print(1)
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p ==1:
            int_reco.append(2)
            print(2)
        elif r_e == 1 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p == 0:
            int_reco.append(3)
            print(3)
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi > 0 and r_p > 1 and r_p < 4:
            int_reco.append(4)
            print(4)
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p > 1 and r_p < 4:
            int_reco.append(5)
            print(5)
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p ==1:
            int_reco.append(6)
            print(6)
        elif r_e == 0 and r_y ==1 and r_m == 0 and r_pi == 0 and r_p == 0:
            int_reco.append(7)
            print(7)
        else:
            int_reco.append(8)
            print(8)
    return int_reco,int_true
    '''
    '''
        if t_e == 1 and t_y ==0:
            int_true.append(0)
            #print('t0')
        elif t_e > 1 and t_y ==0:
            int_true.append(1)
            #print('t1')
        elif t_e == 1 and t_y == 1:
            int_true.append(2)
            #print('t2')
        elif t_e ==0 and t_y ==1:
            int_true.append(3)
            #print('t3')
        elif t_e == 0 and t_y >1:
            int_true.append(4)
            #print('t4')
        elif t_e >1 and t_y >1:
            int_true.append(5)
            #print('t5')
        elif t_e ==0 and t_y ==0 and (t_m > 0 or t_pi > 0 or t_p >0):
            int_true.append(6)
            #print('t6')
        elif t_e == 0 and t_y ==0 and t_m == 0 and t_pi == 0 and t_p ==0:
            int_true.append(7)
            print('t7')
        elif t_none > 0:
            int_true.append(8)
            #print('t8')
        else:
            int_true.append(9)
            #print('t9')
            
        if r_e == 1 and r_y ==0:
            int_reco.append(0)
            print(0)
        elif r_e >1 and r_y ==0:
            int_reco.append(1)
            print(1)
        elif r_e == 1 and r_y ==1:
            int_reco.append(2)
            print(2)
        elif r_e == 0 and r_y ==1:
            int_reco.append(3)
            print(3)
        elif r_e == 0 and r_y >1:
            int_reco.append(4)
            print(4)
        elif r_e > 1 and r_y >1:
            int_reco.append(5)
            print(5)
        elif r_e == 0 and r_y ==0:
            int_reco.append(6)
            print(6)
        elif r_e == 0 and r_y ==0 and r_m == 0 and r_pi == 0 and r_p ==0:
            int_reco.append(7)
            print(7)
        elif r_none >0:
            int_reco.append(8)
            print(8)
        else:
            int_reco.append(9)
            print(9)
    return int_reco,int_true
    '''
        #print("1e0p:", zero, "1e1p:", one,"1enp:", two, "1e1pi:", three, "1e1pi1p:", four,"1p:",five,"1m:",six,"1m1p",seven)
    #return int_reco,int_true
iterations = 500
pred_pid =[]
pred_type = []
true_pid =[]
true_type= []
diff = []
total_r_1e=[]
total_r_1e1p = []
total_r_1enp = []
total_r_1e1pi = []
total_r_1e1pi1p = []
total_r_1p = []
total_r_1m = []
total_r_1m1p = []

total_t_1e=[]
total_t_1e1p = []
total_t_1enp = []
total_t_1e1pi = []
total_t_1e1pi1p = []
total_t_1p = []
total_t_1m = []
total_t_1m1p = []

total_r_1e =[]
total_r_1y = []
total_r_2y = []
other_r = []

total_t_1e =[]
total_t_1y = []
total_t_2y = []
other_t = []





for iteration in range(iterations):
    data,result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        truth_particles = evaluator.get_true_particles(entry, only_primaries=True)
        particles = evaluator.get_particles(entry, only_primaries=True)
        #interact = evaluator.get_true_interactions(entry)
        #if interact ==[]:
        #    continue
        interactions = evaluator.match_interactions(entry)
        #print(interactions)
        matched_particles = evaluator.match_particles(entry, only_primaries=True, mode = 'true_to_pred')
        true_primaries = evaluator.get_true_particles(entry, only_primaries=True)
        #print(matched_particles)
        number = inter_type_loader(matched_particles,interactions,true_primaries)
        pred_type.extend(number[0])
        true_type.extend(number[1])
pred_type = np.array(pred_type)
true_type = np.array(true_type)
        
df = pd.DataFrame({ 'pred_type':pred_type,'true_type':true_type})
df.to_csv("/sdf/home/d/dcarber/pid_csv/int_type.csv")
#print("Total 1e:", total_1e,"Total 1e1p:",total_1e1p,"Total 1enp:", total_1enp,"Total 1e1pi:",total_1e1pi, "Total 1e1pi1p:",total_1e1pi1p, "Total 1p:", total_1p,"Total 1m:",total_1m,"Total 1m1p:", total_1m1p)

# In[ ]:


'''import matplotlib.pyplot as plt
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

plt.hist(diff range=[-1000, 2000],bins=75,label = "Energy_deopsit" )
#plt.hist(collect_reco_neut_energy, range=[-10, 2000], alpha = .6,bins=75,label = "Depo_MeV")
#plt.hist(deposits, range=[0, 300],alpha = 1,bins=100, label = "Sum")
plt.legend(loc = "upper right")
plt.xlabel("Neutrino energy [MeV]")
plt.ylabel("Events")
plt.title("NuMI Nue sample")

'''
# In[ ]:


def plot_confusion_matrix(cm, cm_total,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                         xlabel='Predicted label',
                         ylabel='True label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #xticklabels = ['Shower Frag', 'Track', 'Michel', 'Delta', 'Low E']
    #xticklabels = ['1$e$', '1$\gamma$','2$\gamma$','2$e$','3$/gamma$','Other']
    #xticklabels = ['1$p$', 'n$p$','1$\pi$','n$\pi$','n$p$n$/pi$','n$e$','No part','None','Other']
    #xticklabels = ['1$e$N$p$M$\pi$','1$e$N$p$0$\pi$','1$e$1$p$','1$e$', '1$\gamma$N$p$M$\pi$','1$\gamma$N$p$0$\pi$','1$\gamma$1$p$','1$\gamma$','Other']
    xticklabels = ['1$e$', '1$e$1$p$', '1$e$n$p$', '1$e$1$\pi$', '1$e$1$p$1$\pi$','1$p$','1$\gamma$','1$\gamma$1$p$','1$\gamma$n$p$','1$\gamma$1$\pi$','Other']
    #xticklabels = ['1$e$0$\gamma$X', 'n$e$0$\gamma$X', '1$e$1$\gamma$X', '0$e$1$\gamma$X', '0$e$n$\gamma$X','n$e$n$\gamma$X','0$e$0$\gamma$X','nothing','No True','Other']
    if cm.shape[1] > 20:
        xticklabels.append('Ghost')
    #     if cm.shape[1] > 5:
    #         xticklabels.append('Ghost')

    fig, ax = plt.subplots()
    print(cm.shape[0],cm.shape[1])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    #labels = ['Shower Frag', 'Track', 'Michel', 'Delta', 'Low E']
    #labels = ['1$e$', '1$\gamma$','2$\gamma$','2$e$','3$/gamma$','Other']
    #labels = ['1$e$N$p$M$\pi$','1$e$N$p$0$\pi$','1$e$1$p$','1$e$', '1$\gamma$N$p$M$\pi$','1$\gamma$N$p$0$\pi$','1$\gamma$1$p$','1$\gamma$','Other']
    #labels = ['1$p$', 'n$p$','1$\pi$','n$\pi$','n$p$n$/pi$','n$e$','No part','None','Other']
    #labels = ['1$e$0$\gamma$X', 'n$e$0$\gamma$X', '1$e$1$\gamma$X', '0$e$1$\gamma$X', '0$e$n$\gamma$X','n$e$n$\gamma$X','0$e$0$\gamma$X','nothing','No True','Other']
    labels = ['1$e$', '1$e$1$p$', '1$e$n$p$', '1$e$1$\pi$', '1$e$1$p$1$\pi$','1$p$','1$\gamma$','1$\gamma$1$p$','1$\gamma$n$p$','1$\gamma$1$\pi$','Other']
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=xticklabels, yticklabels=labels,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel,
           ylim=(-0.5, cm.shape[0]-0.5))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt),
                    ha="center", va="top",
                    color="white" if cm[i, j] > thresh else "black")
            ax.text(j, i, format(cm_total[i, j], 'd'),
                    ha="center", va="bottom",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


from numpy.linalg import norm
def confusion_matrix(kinematics, num_classes):
    x = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        for c2 in range(num_classes):
            x[c][c2] =  np.count_nonzero((kinematics['true_type'] == c) & (kinematics['pred_type'] == c2) )
    print(x)
        #x[c][-1] = np.nansum(metrics['num_true_pix_class%d' % c]-metrics['num_true_deghost_pix_class%d' % c])
    return x / x.sum(axis=1, keepdims=True),x


# In[ ]:


METRICS_FOLDER = '/sdf/home/d/dcarber/pid_csv/'
kinematics = pd.read_csv(os.path.join(METRICS_FOLDER, "int_type.csv"))

print(kinematics)
seaborn.set(style="white", font_scale=1)
conf=confusion_matrix(kinematics, 11)
plot_confusion_matrix(np.array(conf[0]*100),np.array(conf[1],dtype = np.int32),normalize =True)
plt.savefig("confusion_matrix.png", transparent=True)
seaborn.set(rc={
    'figure.figsize':(150, 100),
})
seaborn.set_context('talk')


# In[ ]:
