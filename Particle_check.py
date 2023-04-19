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
def pid_type_loader(matched_particles,interact,true_primaries):
    p_pid = []
    p_type = []
    t_pid = []
    t_type = []
    size_diff = []
    particles = []
    true_particles = []
    for m in matched_particles: #Seperates particles and their matched true particle
        particles.append(m[0])
        true_particles.append(m[1])
    '''for i in range(len(interact)):
        inter = interact[i].id
        e = 0
        p = 0
        o =0
        for pri in true_primaries:
            if pri.interaction_id ==inter:
                if pri.pid == 1:
                    print("e")
                    e +=1
                if pri.pid == 4:
                    print("p")
                    p+=1
                else:
                    print("Not e or p")
                    o+=1
        if p !=1 or e !=1 or o > 0:
            print("No 1e1p", e, p, o)
            continue
        print("1e1p Yay")
        for l in range(len(matched_particles)):
            if true_particles[l] == None: #Gets rid of any particle that has no matched true particle
                t_pid.append(5)
                p_pid.append(particles[l].pid)
                t_type.append(6)
                p_type.append(particles[l].semantic_type)
                continue

            #if inter == true_particles[l].interaction_id:
            diff_size = particles[l].size - true_particles[l].size
            size_diff.append(diff_size)
            p_pid.append(particles[l].pid)
            #print(particles[l].pid,particles[l])
            p_type.append(particles[l].semantic_type)
        
            t_pid.append(true_particles[l].pid)
            #print("True particles:/n",true_particles[l].pid,"Particles:/n",particles[l])
        
            t_type.append(true_particles[l].semantic_type)'''
    for l in range(len(matched_particles)):
        if true_particles[l] == None: #Gets rid of any particle that has no matched true particle
            t_pid.append(5)
            p_pid.append(particles[l].pid)
            t_type.append(6)
            p_type.append(particles[l].semantic_type)
            continue

        #if inter == true_particles[l].interaction_id:
        diff_size = particles[l].size - true_particles[l].size
        size_diff.append(diff_size)
        p_pid.append(particles[l].pid)
        #print(particles[l].pid,particles[l])
        p_type.append(particles[l].semantic_type)
        
        t_pid.append(true_particles[l].pid)
        #print("True particles:/n",true_particles[l].pid,"Particles:/n",particles[l])
        
        t_type.append(true_particles[l].semantic_type)
    return p_pid, p_type,t_pid,t_type, size_diff
iterations = 100
pred_pid =[]
pred_type = []
true_pid =[]
true_type= []
diff = []
event=0
total_t_p = 0
total_t_e = 0
total_p_p = 0
total_p_e = 0
total_p_m = 0
total_t_m = 0
total_p_pi = 0
total_t_pi = 0
total_p_pr = 0
total_t_pr = 0
for iteration in range(iterations):
    data,result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        truth_particles = evaluator.get_true_particles(entry, only_primaries=True)
        particles = evaluator.get_particles(entry, only_primaries=True)
        interact = evaluator.get_true_interactions(entry)
        #if interact ==[]:
        #    continue
        #interactions = evaluator.match_interactions(entry)
        #print(interactions)
        matched_particles = evaluator.match_particles(entry, only_primaries=True, mode = 'true_to_pred')
        true_primaries = evaluator.get_true_particles(entry, only_primaries=True)
        #print(matched_particles)
        number = particles_counter(truth_particles,particles)
        if number[1]>number[6]:
            event+=1
            print("A lot of photons! Batch ID: {}, Index: {}".format(entry, index))
        total_p_e += number[0]
        total_p_p += number[1]
        total_p_m += number[2]
        total_p_pi += number[3]
        total_p_pr += number[4]
        
        total_t_e += number[5]
        total_t_p += number[6]
        total_t_m += number[7]
        total_t_pi += number[8]
        total_t_pr += number[9]
        
        #print(true_particles)
        pt = pid_type_loader(matched_particles,interact,true_primaries)
        pred_pid.extend(pt[0])
        pred_type.extend(pt[1])
        true_pid.extend(pt[2])
        true_type.extend(pt[3])
        diff.extend(pt[4])
        df = pd.DataFrame({'pred_pid': pred_pid, 'pred_type':pred_type,'true_pid':true_pid,'true_type':true_type})
df.to_csv("/sdf/home/d/dcarber/pid_csv/pid_type.csv")
print("Total pred electrons:", total_p_e,"Total pred photons:",total_p_p,"Total true electrons:", total_t_e,"Total true photons:",total_t_p, "Total pred muons:",total_p_m, "Total true muons:", total_t_m,"Total pred pions:",total_p_pi,"Total true pions:", total_t_pi,"Total pred protons:",total_p_pr,"Total true protons:", total_t_pr)
print("events", event)
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


def plot_confusion_matrix(cm,
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

    xticklabels = ['Photon', '$e$', '$\mu$', '$\pi$', 'Proton','None']
    if cm.shape[1] > 6:
        xticklabels.append('Ghost')
    #     if cm.shape[1] > 5:
    #         xticklabels.append('Ghost')

    fig, ax = plt.subplots()
    print(cm.shape)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    #labels = ['Shower Frag', 'Track', 'Michel', 'Delta', 'Low E']
    labels = ['Photon', '$e$', '$\mu$', '$\pi$', 'Proton','None']
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
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


from numpy.linalg import norm
def confusion_matrix(kinematics, num_classes):
    x = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        for c2 in range(num_classes):
            x[c][c2] =  np.count_nonzero((kinematics['true_pid'] == c) & (kinematics['pred_pid'] == c2) )
        #x[c][-1] = np.nansum(metrics['num_true_pix_class%d' % c]-metrics['num_true_deghost_pix_class%d' % c])
    return x / x.sum(axis=1, keepdims=True)


# In[ ]:


METRICS_FOLDER = '/sdf/home/d/dcarber/pid_csv/'
kinematics = pd.read_csv(os.path.join(METRICS_FOLDER, "pid_type.csv"))

seaborn.set(style="white", font_scale=2.5)
plot_confusion_matrix(np.array(confusion_matrix(kinematics, 6)*100),normalize =True)
plt.savefig("confusion_matrix.png", transparent=True)
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')


# In[ ]:
