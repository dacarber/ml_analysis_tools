'''
True shower information that has a contained mask
'''

import os, sys
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME')#%s/lartpc_mlreco3d' % os.environ.get('HOME') #'/sdf/group/neutrino/drielsma/me/lartpc_mlreco3d'
DATA_DIR = '/sdf/home/d/dcarber/DATADIR'
# Set software directory
sys.path.append(SOFTWARE_DIR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')
print(SOFTWARE_DIR)
from pprint import pprint
def column(matrix, i):
    return [row[i] for row in matrix]
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
import plotly.express as px
from mlreco.main_funcs import process_config, prepare
import warnings, yaml
warnings.filterwarnings('ignore')

cfg = yaml.load(open('%s/inference_full_chain_020223.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
process_config(cfg, verbose=False)
# prepare function configures necessary "handlers"
hs = prepare(cfg)
dataset = hs.data_io_iter
data, result = hs.trainer.forward(dataset)
from analysis.classes.ui import FullChainEvaluator
# Only run this cell once!
evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
print(evaluator)
entry = 9 # Batch ID for current sample
print("Batch ID = ", evaluator.index[entry])
print("Index", evaluator.index)
particles = evaluator.get_particles(entry, only_primaries=True)
true_particles = evaluator.get_true_particles(entry, only_primaries=False)

min_segment_size = 3 # in voxels
radius = 10 # in voxels
thickness = 3


pca = PCA(n_components=2)
def containment_mask(particle):
    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]
    contained_mask = []
    for v in particle.coords_noghost:
        absolute_pos = v * size_voxel_x + min_x
        if absolute_pos[0] <= -359.63 or (absolute_pos[0] >= -60.8 and absolute_pos[0] <= 60.8) or  absolute_pos[0] >= 359.63:
            contained_mask.append(False)
            continue
        elif absolute_pos[1] <= -181.86 or absolute_pos[1] >= 134.96:
            contained_mask.append(False)
            continue
        elif absolute_pos[2] <= -895.95 or absolute_pos[2] >= 894.95:
            contained_mask.append(False)
            continue
        else:
            contained_mask.append(True)
    return contained_mask
        
        
def compute_shower_dqdx(true_particles, r=10, min_segment_size=3):
    '''
    Inputs:
        -matched_particles(list of matched particles)
        -matched_interaction(list of matched interactions)
        -radius of the shower dE/dx
        -minimum segemtne size for the showers
    Returns:
        -out: list of computed dE/dx for each particle
        -out_electron: list of computed dE/dx for true electrons
        -out_photon: list of computed dE/dx for true photons
        -out_true: list of true dQ/dx for each particle
        -out_total: list of true total energy for each particle
        -out_true_electron: list of true dE/dx for true electrons
        -out_true_photon: list of true dE/dx for true photons
        -out_diff: list of difference between the true vs reco dE/dx
    '''
    '''List of arrays and declaring variables'''
    out= []
    out_electron = []
    out_photon = []
    out_true = []
    out_total =[]
    out_true_photon = []
    out_true_electron = []
    out_true_total = []
    out_total_e =[]
    out_total_p =[]
    out_true_total_e =[]
    out_true_total_p = []
    out_con_total =[]
    out_con_total_p = []
    out_con_total_e = []
    out_x =[]
    out_y =[]
    out_z =[]

    
    
    
    '''List of matched interaction vertex'''
    #interaction_vertexs = matched_interaction[0][0].vertex
    #interaction_vertex = np.array(interaction_vertexs)
    '''Grabbing the absolute coordinates within the detector'''
    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]
    tdrift = 0

    for i in range(len(true_particles)): #Loops over all particles in the matched particle list
        '''Makes sure the particles are what we want'''
        if true_particles[i] == None: #Gets rid of any particle that has no matched true particle
            continue
        if(true_particles[i].pid > 1): #Selects either photon or electron particles
            continue
        if(true_particles[i].semantic_type > 1):#Selects fragment showers
            continue
        assert true_particles[i].is_primary #Checks if the particle is a primary particle
        if (true_particles[i].startpoint < 0).any():
            continue   

        
        '''If the PPN startpoint is within a certain distance from the closest point to vertex it will choose the PPN start point'''
        
       
        
        '''Grabs true particle information if the true points are within the assigned radius '''
        true_startpoint = true_particles[i].startpoint
        true_dist = cdist(true_particles[i].points, true_startpoint.reshape(1,-1))
        true_mask = true_dist.squeeze() < r #Are the points' distance from start point within selected radius
        '''Finds a mask for points that are within the radius and outside of your inner radius'''
        
                

       

        
        true_selected_points = true_particles[i].points[true_mask]#Grabs true points that are within the radius
       
        if true_selected_points.shape[0] < 2:# Makes sure there are enough points
            continue
        '''Finds the drift time to the corresponding anode to calculate the calibration from ADC to MeV'''
        absolute_pos = true_selected_points * size_voxel_x + min_x
        out_of_detector = False
        for pt in range(len(absolute_pos)):
            if absolute_pos[pt][0] >= -359.63 and absolute_pos[pt][0] <= -210.29:
                anode_dist = abs(absolute_pos[pt][0] +359.63)
                tdrift =  10*anode_dist/ (1.6)
            if absolute_pos[pt][0] >= -210.14 and absolute_pos[pt][0] <= -60.8:
                anode_dist = abs(absolute_pos[pt][0] +60.8)
                tdrift =  10*anode_dist/ (1.6)
            if absolute_pos[pt][0] <= 210.14 and absolute_pos[pt][0] >= 60.8:
                anode_dist = abs(absolute_pos[pt][0] -60.8)
                tdrift =  10*anode_dist/ (1.6)
            if absolute_pos[pt][0] <= 359.63 and absolute_pos[pt][0] >= 210.14:
                anode_dist = abs(absolute_pos[pt][0] -359.63)
                tdrift =  10*anode_dist/ (1.6)
            if absolute_pos[pt][0] >= 359.63 and absolute_pos[pt][0] <= -359.63:
                out_of_detector = True
        if out_of_detector == True:
            continue
            
        con_mask = containment_mask(true_particles[i])
        '''Grabs dq and dx information'''
        true_proj = pca.fit_transform(true_selected_points)
        
        true_dx = (true_proj[:,0].max() - true_proj[:,0].min())
        if true_dx < min_segment_size:
            continue
        true_dx = (true_dx)*.3 #Converts voxels into cm
        
        out_x.append(true_particles[i].startpoint[0])
        out_y.append(true_particles[i].startpoint[1])
        out_z.append(true_particles[i].startpoint[2])
        
        contained_energy = np.sum(true_particles[i].depositions_noghost[con_mask])
        
        dq_true = np.sum(true_particles[i].depositions_noghost[true_mask])*(23.6*10**(-6))*(85.25)*np.exp(tdrift/3000)*0.66**(-1)  # Grabs true dE 
        
        true_total_energy = np.sum(true_particles[i].depositions_noghost)#*(23.6*10**(-6))*(85.25)*np.exp(tdrift/3000)*0.66**(-1) 
        '''Grabs the output information'''
        #out_diff.append(diff)
        out_true_total.append(true_total_energy)
        out_true.append(dq_true/true_dx)
        out_con_total.append(contained_energy)
        '''Grabs the dE/dx info for photon and electrons'''
        
        if true_particles[i].pid == 0:   
            out_true_photon.append(dq_true/true_dx)
            out_true_total_p.append(true_total_energy)
            out_con_total_p.append(contained_energy)
        if true_particles[i].pid == 1:
            out_true_electron.append(dq_true/true_dx)
            out_true_total_e.append(true_total_energy)
            out_con_total_e.append(contained_energy)
    
    return out_true, out_true_photon, out_true_electron, out_true_total,out_true_total_e, out_true_total_p, out_con_total,out_con_total_p,out_con_total_e, out_x, out_y, out_z

iterations = 400

collect_dqdx_true = []
collect_dqdx_true_electrons = []
collect_dqdx_true_photons = []
collect_true_total =[]
collect_true_total_e =[]
collect_true_total_p =[]
collect_con_total = []
collect_con_total_p = []
collect_con_total_e = []
collect_out_x = []
collect_out_y = []
collect_out_z = []
for iteration in range(iterations):
    data, result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        #matched_interaction = evaluator.match_interactions(entry)
        true_particles = evaluator.get_true_particles(entry, only_primaries=True)
        dqdx = compute_shower_dqdx(true_particles, r=radius, min_segment_size=min_segment_size)
        collect_dqdx_true.extend(dqdx[0])
        collect_dqdx_true_photons.extend(dqdx[1])
        collect_dqdx_true_electrons.extend(dqdx[2])
        collect_true_total.extend(dqdx[3])
        collect_true_total_e.extend(dqdx[4])
        collect_true_total_p.extend(dqdx[5])
        collect_con_total.extend(dqdx[6])
        collect_con_total_p.extend(dqdx[7])
        collect_con_total_e.extend(dqdx[8])
        collect_out_x.extend(dqdx[9])
        collect_out_y.extend(dqdx[10])
        collect_out_z.extend(dqdx[11])


collect_dqdx_true = np.array(collect_dqdx_true)
collect_dqdx_true_photons = np.array(collect_dqdx_true_photons)
collect_dqdx_true_electrons = np.array(collect_dqdx_true_electrons)
collect_true_total = np.array(collect_true_total)
collect_true_total_e = np.array(collect_true_total_e)
collect_true_total_p = np.array(collect_true_total_p)
collect_con_total = np.array(collect_con_total)
collect_con_total_p = np.array(collect_con_total_p)
collect_con_total_e = np.array(collect_con_total_e)
collect_out_x = np.array(collect_out_x)
collect_out_y = np.array(collect_out_y)
collect_out_z = np.array(collect_out_z)




                               

pd.DataFrame({'true':collect_dqdx_true}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true.csv')
pd.DataFrame({'true_p':collect_dqdx_true_photons}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true_p.csv')
pd.DataFrame({'true_e':collect_dqdx_true_electrons}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true_e.csv')
pd.DataFrame({'total':collect_true_total}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true_total.csv')
pd.DataFrame({'total_e':collect_true_total_e}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true_total_e.csv')
pd.DataFrame({'total_p':collect_true_total_p}).to_csv(f'~/data/data_r{radius}_NuMI_nue_true_total_p.csv')
pd.DataFrame({'contain':collect_con_total}).to_csv(f'~/data/data_r{radius}_NuMI_nue_con_total.csv')
pd.DataFrame({'contain_p':collect_con_total_p}).to_csv(f'~/data/data_r{radius}_NuMI_nue_con_total_p.csv')
pd.DataFrame({'contain_e':collect_con_total_e}).to_csv(f'~/data/data_r{radius}_NuMI_nue_con_total_e.csv')
pd.DataFrame({'x':collect_out_x}).to_csv(f'~/data/data_r{radius}_NuMI_nue_start_x.csv')
pd.DataFrame({'y':collect_out_y}).to_csv(f'~/data/data_r{radius}_NuMI_nue_start_y.csv')
pd.DataFrame({'z':collect_out_z}).to_csv(f'~/data/data_r{radius}_NuMI_nue_start_z.csv')


print("True Median for electrons: ",np.median(collect_dqdx_true_electrons)," True Median for photons: ", np.median(collect_dqdx_true_photons))
print("True Mean for electrons: " , collect_dqdx_true_electrons.mean()," True Mean for photons: ", collect_dqdx_true_photons.mean())
print("True Total electrons ",len(collect_dqdx_true_electrons),", True Total photons ", len(collect_dqdx_true_photons))



f = open("datafile1.txt", "w")
f.write(f"True Median for electrons: {np.median(collect_dqdx_true_electrons)} True Median for photons:  {np.median(collect_dqdx_true_photons)} True Mean for electrons: {collect_dqdx_true_electrons.mean()} True Mean for photons:  {collect_dqdx_true_photons.mean()} True Total electrons {len(collect_dqdx_true_electrons)} True Total photons {len(collect_dqdx_true_photons)}")
f.close()

