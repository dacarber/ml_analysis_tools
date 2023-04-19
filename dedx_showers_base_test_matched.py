'''
Shower dE/dx base testing code
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

min_segment_size = 3 # in voxels
radius = 10 # in voxels
thickness = 3


pca = PCA(n_components=2)
def compute_shower_dqdx(matched_particles, r=10, min_segment_size=3):
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
    out_photon = []
    out_electron = []
    out_electron_total =[]
    out_photon_total = []
    
    out_true= []
    out_electron_true = []
    out_photon_true = []
    out_total_true =[]
    out_photon_true = []
    out_electron_true = []
    out_electron_total_true =[]
    out_photon_total_true = []
    particles = []
    true_particles = []

    '''List of matched interaction vertex'''
    #interaction_vertexs = matched_interaction[0][0].vertex
    #interaction_vertex = np.array(interaction_vertexs)
    '''Grabbing the absolute coordinates within the detector'''
    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]
    tdrift = 0
    for m in matched_particles: #Seperates particles and their matched true particle
        particles.append(m[0])
        true_particles.append(m[1])
    for i in range(len(matched_particles)): #Loops over all particles in the matched particle list
        '''Makes sure the particles are what we want'''
        if(particles[i].pid > 1): #Selects either photon or electron particles
            continue
        if(particles[i].semantic_type > 1):#Selects fragment showers
            continue
        assert particles[i].is_primary #Checks if the particle is a primary particle
        if (particles[i].startpoint < 0).any():
            continue   
        if true_particles[i] == None: #Gets rid of any particle that has no matched true particle
            '''print("1")
            ppn_prediction = particles[i].startpoint
            dist = cdist(particles[i].points, ppn_prediction.reshape(1,-1))#Distance of each point in particle from the start point
            mask = dist.squeeze() < r
            selected_points = particles[i].points[mask]
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
            proj = pca.fit_transform(selected_points)
            dx = (proj[:,0].max() - proj[:,0].min())
            if dx < min_segment_size:
                print("7")
                continue
            dx = (dx)*.3
            dq = np.sum(particles[i].depositions[mask])*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1)
            dq_true = 0
            total_energy = np.sum(particles[i].depositions)*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1)  #Grabs the total energy of the true particle
            true_total_energy = 0
            out_total.append(total_energy)
            out.append(dq/dx)
            out_true.append(0)
            out_total_true.append(0)
            if particles[i].pid == 0:
                print("8")
                out_photon.append(dq/dx)
                out_photon_true.append(0)
                out_photon_total.append(total_energy)
                out_photon_total_true.append(0)
            if particles[i].pid == 1:
                print("9")
                out_electron.append(dq/dx)
                out_electron_true.append(0)
                out_electron_total.append(total_energy)
                out_electron_total_true.append(0)'''
            #return out, out_total, out_photon, out_electron, out_photon_total,out_electron_total, out_true, out_total_true, out_photon_true, out_electron_true, out_photon_total_true,out_electron_total_true
            continue
        
        point_dist =[]
        ppn_prediction = particles[i].startpoint  #This is if you just want to grab the startpoint assigned from the ML reco
        
        '''Grabs true particle information if the true points are within the assigned radius '''
        true_startpoint = true_particles[i].startpoint
        true_dist = cdist(true_particles[i].points, true_startpoint.reshape(1,-1))
        true_mask = true_dist.squeeze() < r #Are the points' distance from start point within selected radius
        
        dist = cdist(particles[i].points, ppn_prediction.reshape(1,-1))#Distance of each point in particle from the start point'''
        mask = dist.squeeze() < r #Are the points' distance from start point within selected radius
        #total_mask = 
        selected_points = particles[i].points[mask]#Grabs points that are within the radius
        true_selected_points = true_particles[i].points[true_mask]#Grabs true points that are within the radius
        #print(selected_points)
        #print(true_selected_points)
        if selected_points.shape[0] < 2:# Makes sure there are enough points
            continue
        if true_selected_points.shape[0] < 2:# Makes sure there are enough points
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
        true_proj = pca.fit_transform(true_selected_points)
        dx = (proj[:,0].max() - proj[:,0].min())
        if dx < min_segment_size:
            continue
        true_dx = (true_proj[:,0].max() - true_proj[:,0].min())
        
        dx = (dx)*.3 #Converts voxels into cm
        true_dx = (true_dx)*.3 #Converts voxels into cm
        
        #Conversts ADC to MeV using Lane's method
        dq = np.sum(particles[i].depositions[mask])*(23.6*10**(-6))*(85.25)*np.exp(tdrift/3000)*0.66**(-1)#*(23.6*10**(-6))*(93.5)*(1.18)*0.7**(-1) 
        #*(23.6*10**(-6))*(85.25)*np.exp(tdrift/3000)*0.66**(-1) 
        
        #Converts ADC to MeV using the Modified box model
        '''dq = np.sum(particles[i].depositions[mask])*np.exp(tdrift/3000)
        dedx = (np.exp(((dq/dx)*(23.6*10**(-6))*86.83*.212)/(1.383*.5))-.93)/(.212/(1.383*.5))'''

        dq_true = np.sum(true_particles[i].depositions_MeV[true_mask]) # Grabs true dE 
        
        total_energy = np.sum(true_particles[i].depositions_noghost)#*(23.6*10**(-6))*(85.25)*np.exp(tdrift/3000)*0.66**(-1)  #Grabs the total energy of the true particle
        true_total_energy = np.sum(true_particles[i].depositions_MeV)#*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1) #Grabs the total energy of the true particle
        '''Grabs the output information'''
        out_total.append(total_energy)
        out.append(dq)
        out_true.append(dq_true)
        out_total_true.append(true_total_energy)
        '''Grabs the dE/dx info for photon and electrons'''

        if dq_true <10 and dq >10:
            print("High")
        if true_particles[i].pid == 0:
            out_photon.append(dq)
            out_photon_true.append(dq_true)
            out_photon_total.append(total_energy)
            out_photon_total_true.append(true_total_energy)
        if true_particles[i].pid == 1:
            
            out_electron.append(dq)
            out_electron_true.append(dq_true)
            out_electron_total.append(total_energy)
            out_electron_total_true.append(true_total_energy)
    return out, out_total, out_photon, out_electron, out_photon_total,out_electron_total, out_true, out_total_true, out_photon_true, out_electron_true, out_photon_total_true,out_electron_total_true

iterations = 500

collect_dqdx = []
collect_total = []
collect_dqdx_photons = []
collect_dqdx_electrons = []
collect_total_photons = []
collect_total_electrons = []

collect_dqdx_true = []
collect_total_true = []
collect_dqdx_photons_true = []
collect_dqdx_electrons_true = []
collect_total_photons_true = []
collect_total_electrons_true = []


for iteration in range(iterations):
    data, result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        matched_particles = evaluator.match_particles(entry, only_primaries = True,mode = 'pred_to_true')

        dqdx = compute_shower_dqdx(matched_particles, r=radius, min_segment_size=min_segment_size)
        collect_dqdx.extend(dqdx[0])
        collect_total.extend(dqdx[1])
        collect_dqdx_photons.extend(dqdx[2])
        collect_dqdx_electrons.extend(dqdx[3])
        collect_total_photons.extend(dqdx[4])
        collect_total_electrons.extend(dqdx[5])
        
        collect_dqdx_true.extend(dqdx[6])
        collect_total_true.extend(dqdx[7])
        collect_dqdx_photons_true.extend(dqdx[8])
        collect_dqdx_electrons_true.extend(dqdx[9])
        collect_total_photons_true.extend(dqdx[10])
        collect_total_electrons_true.extend(dqdx[11])
        

collect_dqdx = np.array(collect_dqdx)
collect_total = np.array(collect_total)
collect_dqdx_photons = np.array(collect_dqdx_photons)
collect_dqdx_electrons = np.array(collect_dqdx_electrons)
collect_total_photons = np.array(collect_total_photons)
collect_total_electrons = np.array(collect_total_electrons)

collect_dqdx_true = np.array(collect_dqdx_true)
collect_total_true = np.array(collect_total_true)
collect_dqdx_photons_true = np.array(collect_dqdx_photons_true)
collect_dqdx_electrons_true = np.array(collect_dqdx_electrons_true)
collect_total_photons_true = np.array(collect_total_photons_true)
collect_total_electrons_true = np.array(collect_total_electrons_true)


pd.DataFrame({'reco':collect_dqdx}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_reco.csv')
pd.DataFrame({'total':collect_total}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total.csv')
pd.DataFrame({'photons':collect_dqdx_photons}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_photons.csv')
pd.DataFrame({'electrons':collect_dqdx_electrons}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_electrons.csv')
pd.DataFrame({'total_p':collect_total_photons}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total_p.csv')
pd.DataFrame({'total_e':collect_total_electrons}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total_e.csv')

pd.DataFrame({'reco':collect_dqdx_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_reco_true.csv')
pd.DataFrame({'total':collect_total_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total_true.csv')
pd.DataFrame({'photons':collect_dqdx_photons_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_photons_true.csv')
pd.DataFrame({'electrons':collect_dqdx_electrons_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_electrons_true.csv')
pd.DataFrame({'total_p':collect_total_photons_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total_p_true.csv')
pd.DataFrame({'total_e':collect_total_electrons_true}).to_csv(f'~/data/data_r{radius}_matched_numi_nue_total_e_true.csv')


