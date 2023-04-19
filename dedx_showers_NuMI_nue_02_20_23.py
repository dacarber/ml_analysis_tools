'''
Shower dE/dx For NuMI nue sample. Got rid of matching function
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
def compute_shower_dqdx(particles,interaction, r=10, min_segment_size=3):
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
    out_total =[]
    out_total_p =[]
    out_total_e =[]
    out_total_hits = []

    '''List of matched interaction vertex'''
    #interaction_vertexs = interaction.vertex
    #interaction_vertex = np.array(interaction_vertexs)
    '''Grabbing the absolute coordinates within the detector'''
    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]
    tdrift = 0

    for i in range(len(particles)): #Loops over all particles in the matched particle list
        '''Makes sure the particles are what we want'''
        if(particles[i].pid > 1): #Selects either photon or electron particles
            continue
        if(particles[i].semantic_type > 1):#Selects fragment showers
            continue
        if (particles[i].is_primary==False):#Checks if the particle is a primary particle
            continue
        if (particles[i].startpoint < 0).any():
            continue   
        point_dist =[]
        '''for points in particles[i].points: #Finds distance from points in the shower and the vertex of the interaction
            point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
        closest_point = np.argmin(point_dist)#Finds the point with the minimum distance '''
        '''If the PPN startpoint is within a certain distance from the closest point to vertex it will choose the PPN start point'''
        '''if cdist(particles[i].points[closest_point].reshape(1,-1),particles[i].startpoint.reshape(1,-1)) >= 4: 
            for points in particles[i].points:
                point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
            closest_point = np.argmin(point_dist)
            ppn_prediction = particles[i].points[closest_point]
        else:
            ppn_prediction = particles[i].startpoint'''
        ppn_prediction = particles[i].startpoint  #This is if you just want to grab the startpoint assigned from the ML reco
        

        '''Finds a mask for points that are within the radius and outside of your inner radius'''
        
                
        dist = cdist(particles[i].points, ppn_prediction.reshape(1,-1))#Distance of each point in particle from the start point'''
        mask = dist.squeeze() < r #Are the points' distance from start point within selected radius
        '''Finds a mask for points that are within the radius and outside of your inner radius'''
        #mask = []
        #for d in dist:
            #vertex_dist = cdist(ppn_prediction.reshape(1,-1), interaction_vertex.reshape(1,-1))
            #if vertex_dist > 2:
            #    mask = dist.squeeze() < r
            #    break
         #   if d < r and d > 2:
          #      mask.append(True)
         #   else:
         #       mask.append(False)
        selected_points = particles[i].points[mask]#Grabs points that are within the radius
        
        if selected_points.shape[0] < 2:# Makes sure there are enough points
            continue
        '''Finds the drift time to the corresponding anode to calculate the calibration from ADC to MeV'''
        absolute_pos = selected_points * size_voxel_x + min_x
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
        '''Grabs dq and dx information'''
        proj = pca.fit_transform(selected_points)
        dx = (proj[:,0].max() - proj[:,0].min())
        if dx < min_segment_size:
            continue
        
        dx = (dx)*.3 #Converts voxels into cm
        
        #Conversts ADC to MeV using Lane's method
        dq = np.sum(particles[i].depositions[mask])*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1) 
        
        #Converts ADC to MeV using the Modified box model
        '''dq = np.sum(particles[i].depositions[mask])*np.exp(tdrift/3000)
        dedx = (np.exp(((dq/dx)*(23.6*10**(-6))*86.83*.212)/(1.383*.5))-.93)/(.212/(1.383*.5))'''
        
        
        total_energy = np.sum(particles[i].depositions)*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1)  #Grabs the total energy of the true particle
        
        '''Grabs the output information'''
        #out_diff.append(diff)
        out_total.append(total_energy)
        out.append(dq/dx)
        '''Grabs the dE/dx info for photon and electrons'''
        
        if particles[i].pid == 0:        
            out_photon.append(dq/dx)
            out_total_p.append(total_energy)
        if particles[i].pid == 1:
            out_electron.append(dq/dx)
            out_total_e.append(total_energy)
        out_total_hits.append(np.size(particles[i].depositions))
    
    return out_photon, out_electron, out, out_total, out_total_p, out_total_e, out_total_hits
iterations = 140

collect_dqdx_electrons = []
collect_dqdx_photons = []
collect_dqdx = []
collect_dqdx_total = []
collect_dqdx_total_p = []
collect_dqdx_total_e = []
collect_dqdx_total_hits = []

for iteration in range(iterations):
    data, result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        particles = evaluator.get_particles(entry, only_primaries = True)
        interaction = evaluator.get_interactions(entry)

        dqdx = compute_shower_dqdx(particles,interaction, r=radius, min_segment_size=min_segment_size)
        collect_dqdx_photons.extend(dqdx[0])
        collect_dqdx_electrons.extend(dqdx[1])
        collect_dqdx.extend(dqdx[2])
        collect_dqdx_total.extend(dqdx[3])
        collect_dqdx_total_p.extend(dqdx[4])
        collect_dqdx_total_e.extend(dqdx[5])
        collect_dqdx_total_hits.extend(dqdx[6])

collect_dqdx_electrons = np.array(collect_dqdx_electrons)
collect_dqdx_photons = np.array(collect_dqdx_photons)
collect_dqdx = np.array(collect_dqdx)
collect_dqdx_total = np.array(collect_dqdx_total)
collect_dqdx_total_p = np.array(collect_dqdx_total_p)
collect_dqdx_total_e = np.array(collect_dqdx_total_e)
collect_dqdx_total_hits = np.array(collect_dqdx_total_hits)


pd.DataFrame({'photons':collect_dqdx_photons}).to_csv(f'~/data/data_r{radius}_NuMI_nue_photons.csv')
pd.DataFrame({'electrons':collect_dqdx_electrons}).to_csv(f'~/data/data_r{radius}_NuMI_nue_electrons.csv')
pd.DataFrame({'reco':collect_dqdx}).to_csv(f'~/data/data_r{radius}_NuMI_nue_reco.csv')
pd.DataFrame({'total':collect_dqdx_total}).to_csv(f'~/data/data_r{radius}_NuMI_nue_total.csv')
pd.DataFrame({'total_p':collect_dqdx_total_p}).to_csv(f'~/data/data_r{radius}_NuMI_nue_total_p.csv')
pd.DataFrame({'total_e':collect_dqdx_total_e}).to_csv(f'~/data/data_r{radius}_NuMI_nue_total_e.csv')
pd.DataFrame({'total_hits':collect_dqdx_total_hits}).to_csv(f'~/data/data_r{radius}_NuMI_nue_total_hits.csv')

print("Median for electrons: ",np.median(collect_dqdx_electrons)," Median for photons: ", np.median(collect_dqdx_photons))
print("Mean for electrons: " , collect_dqdx_electrons.mean()," Mean for photons: ", collect_dqdx_photons.mean())
print("Total electrons ",len(collect_dqdx_electrons),", Total photons ", len(collect_dqdx_photons))
#print("Mean diff: ", np.mean(collect_diff), " Median diff: ", np.median(collect_diff))


f = open("datafile1.txt", "w")
f.write(f"Median for electrons: {np.median(collect_dqdx_electrons)} Median for photons:  {np.median(collect_dqdx_photons)} Mean for electrons: {collect_dqdx_electrons.mean()} Mean for photons:  {collect_dqdx_photons.mean()} Total electrons {len(collect_dqdx_electrons)} Total photons {len(collect_dqdx_photons)}")
f.close()

