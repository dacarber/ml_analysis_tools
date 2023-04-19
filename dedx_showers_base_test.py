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

cfg = yaml.load(open('%s/inference_full_volume_112022.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
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
def compute_shower_dqdx(matched_particles,matched_interaction, r=10, min_segment_size=3):
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
    depo1 = []
    depo =[]
    out_diff = []
    particles = []
    true_particles = []
    last_id = []
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
        if true_particles[i] == None: #Gets rid of any particle that has no matched true particle
            continue
        if(particles[i].pid > 1): #Selects either photon or electron particles
            continue
        if(particles[i].semantic_type > 1):#Selects fragment showers
            continue
        assert particles[i].is_primary #Checks if the particle is a primary particle
        if (particles[i].startpoint < 0).any():
            continue   
        point_dist =[]
        for points in particles[i].points: #Finds distance from points in the shower and the vertex of the interaction
            point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
        closest_point = np.argmin(point_dist)#Finds the point with the minimum distance 
        '''If the PPN startpoint is within a certain distance from the closest point to vertex it will choose the PPN start point'''
        '''if cdist(particles[i].points[closest_point].reshape(1,-1),particles[i].startpoint.reshape(1,-1)) >= 4: 
            for points in particles[i].points:
                point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
            closest_point = np.argmin(point_dist)
            ppn_prediction = particles[i].points[closest_point]
        else:
            ppn_prediction = particles[i].startpoint'''
        ppn_prediction = particles[i].startpoint  #This is if you just want to grab the startpoint assigned from the ML reco
        
        '''Grabs true particle information if the true points are within the assigned radius '''
        true_startpoint = true_particles[i].startpoint
        true_dist = cdist(true_particles[i].points, true_startpoint.reshape(1,-1))
        true_mask = true_dist.squeeze() < r #Are the points' distance from start point within selected radius
        '''Finds a mask for points that are within the radius and outside of your inner radius'''
        '''true_mask = []
        for td in true_dist:
            if td < r  and td > 2:
                true_mask.append(True)
            else:
                true_mask.append(False)'''
                
        dist = cdist(particles[i].points, ppn_prediction.reshape(1,-1))#Distance of each point in particle from the start point'''

        mask = dist.squeeze() < r #Are the points' distance from start point within selected radius
        '''Finds a mask for points that are within the radius and outside of your inner radius'''
        '''mask = []
        for d in dist:
            vertex_dist = cdist(ppn_prediction.reshape(1,-1), interaction_vertex.reshape(1,-1))
            if vertex_dist > 2:
                mask = dist.squeeze() < r
                break
            if d < r and d > 2:
                mask.append(True)
            else:
                mask.append(False)'''
        selected_points = particles[i].points[mask]#Grabs points that are within the radius
        true_selected_points = true_particles[i].points[true_mask]#Grabs true points that are within the radius
        new_dist = cdist(selected_points,ppn_prediction.reshape(1,-1))
        min_dist = min(new_dist)
        max_dist = max(new_dist)
        min_point = selected_point[min_dist]
        max_point = selected_point[max_dist]
        phi = np.arctan((max_point[0] - min_point[0])/(max_point[2] - min_point[2]))
        theta = np.arctan((max_point[1] - min_point[1])/(max_point[2] - min_point[2]))
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
        dq = np.sum(particles[i].depositions[mask])*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1) 
        
        #Converts ADC to MeV using the Modified box model
        '''dq = np.sum(particles[i].depositions[mask])*np.exp(tdrift/3000)
        dedx = (np.exp(((dq/dx)*(23.6*10**(-6))*86.83*.212)/(1.383*.5))-.93)/(.212/(1.383*.5))'''

        dq_true = np.sum(true_particles[i].depositions_MeV[true_mask]) # Grabs true dE 
        
        total_energy = np.sum(true_particles[i].depositions_MeV) #Grabs the total energy of the true particle
        diff = (dq/dx - dq_true/true_dx)/(dq_true/true_dx) # Finds the difference between reco vs true
        
        '''Grabs the output information'''
        #out_diff.append(diff)
        out_total.append(true_particles[i].size)
        #print(true_particles[i].size)
        out.append(dq/dx)
        out_true.append(dq_true/true_dx)
        '''Grabs the dE/dx info for photon and electrons'''
        
        if true_particles[i].pid == 0:
            if dq/dx < 3.5:
                print(particles[i].id, true_particles[i].id)
                out_diff.append(true_particles[i].size)
            #if (true_particles[i].id == ids for ids in last_id):
            #    total_multiples += 1
                #print(f"Multiples:{true_particles[i].id}")
            #last_id.append(true_particles[i].id)
                
            out_photon.append(dq/dx)
            out_true_photon.append(dq_true/true_dx)
        if true_particles[i].pid == 1:
            out_electron.append(dq/dx)
            out_true_electron.append(dq_true/true_dx)
    return out_photon, out_electron, out, out_true, out_true_photon, out_true_electron, out_diff,out_total

iterations = 50

collect_dqdx_electrons = []
collect_dqdx_photons = []
collect_dqdx = []
collect_dqdx_true = []
collect_dqdx_total = []
collect_dqdx_true_electrons = []
collect_dqdx_true_photons = []
collect_diff = []
depo1 = [] 
depo = []
collect_startpoint = []
collect_interaction = []
low_e = []
sel_e = 0
parts_out = 0
multiples = 0
false_part = 0
for iteration in range(iterations):
    data, result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        print("Batch ID: {}, Index: {}".format(entry, index))
        matched_particles = evaluator.match_particles(entry, only_primaries = True,mode = 'pred_to_true')
        matched_interaction = evaluator.match_interactions(entry)
        if matched_interaction == []:
            continue

        dqdx = compute_shower_dqdx(matched_particles,matched_interaction, r=radius, min_segment_size=min_segment_size)
        collect_dqdx_photons.extend(dqdx[0])
        collect_dqdx_electrons.extend(dqdx[1])
        collect_dqdx.extend(dqdx[2])
        collect_dqdx_true.extend(dqdx[3])
        #collect_diff.extend(dqdx[6])
        depo1.extend(dqdx[4])
        depo.extend(dqdx[5])
        low_e.extend(dqdx[6])
        collect_dqdx_total.extend(dqdx[7])


        
#print(f"Mutiple total: {multiples}")
print(f"False total: {false_part}")

collect_dqdx_electrons = np.array(collect_dqdx_electrons)
collect_dqdx_photons = np.array(collect_dqdx_photons)
collect_dqdx = np.array(collect_dqdx)
collect_dqdx_true = np.array(collect_dqdx_true)
collect_diff = np.array(collect_diff)
depo1 = np.array(depo1)
depo = np.array(depo)
low_e = np.array(low_e)
collect_dqdx_total = np.array(collect_dqdx_total)

pd.DataFrame({'photons':collect_dqdx_photons}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_photons_test.csv')
pd.DataFrame({'electrons':collect_dqdx_electrons}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_electrons_test.csv')
pd.DataFrame({'reco':collect_dqdx}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_reco_test.csv')
pd.DataFrame({'true':collect_dqdx_true}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_true_test.csv')
pd.DataFrame({'true_p':depo1}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_true_photons_test.csv')
pd.DataFrame({'true_e':depo}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_true_electrons_test.csv')
pd.DataFrame({'low_e':low_e}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_low_e_test.csv')
pd.DataFrame({'total_e':collect_dqdx_total}).to_csv(f'~/data/data_r{radius}_c2_lane_no_circle_v2_6_total_e_test.csv')
print("Median for electrons: ",np.median(collect_dqdx_electrons)," Median for photons: ", np.median(collect_dqdx_photons))
print("Mean for electrons: " , collect_dqdx_electrons.mean()," Mean for photons: ", collect_dqdx_photons.mean())
print("Total electrons ",len(collect_dqdx_electrons),", Total photons ", len(collect_dqdx_photons))
print("Mean diff: ", np.mean(collect_diff), " Median diff: ", np.median(collect_diff))


f = open("datafile1.txt", "w")
f.write(f"Median for electrons: {np.median(collect_dqdx_electrons)} Median for photons:  {np.median(collect_dqdx_photons)} Mean for electrons: {collect_dqdx_electrons.mean()} Mean for photons:  {collect_dqdx_photons.mean()} Total electrons {len(collect_dqdx_electrons)} Total photons {len(collect_dqdx_photons)} Mean diff: {np.mean(collect_diff)} Median diff: {np.median(collect_diff)} RMS diff: {rms_diff} Total diff:  {total_diff}")
f.close()

seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')

plt.hist(collect_dqdx_electrons, range=[0, 10],bins=50, label = "Electrons")
plt.hist(collect_dqdx_photons, range=[0, 10], alpha = .6,bins=50,label = "Photons")
plt.hist(collect_dqdx, range=[0, 10],alpha = .2,bins=50, label = "Sum")
plt.legend(loc = "upper right")
plt.xlabel("dE/dx [MeV/cm]")
plt.ylabel("Showers")
plt.title("Shower dQdx Energy fixed ADC to MeV conversion Out of volume cut ")
plt.show()
plt.savefig('dedx_4000_events.png')

diff = []
std =[]
x_bins = 100
num =0
diff = (np.subtract(collect_dqdx,collect_dqdx_true))/collect_dqdx_true
fig = plt.subplots(figsize =(10, 7))

for i in range(x_bins):
    values = []
    for t in range(len(collect_dqdx_true)):
        num =collect_dqdx_true[t]
        if num>=i*10 and num<(i+1)*10:
            values.append(diff[t])
    std.append(np.std(values))
plot1 = plt.hist2d(collect_dqdx_true, diff,range = [[0, 10], [-1, 1]], bins = [x_bins,100],cmap=plt.cm.jet)   
#plt.title("True Shower Start vs. Reco Shower Start")
plt.xlabel("True dE/dx [MeV/cm]")
plt.ylabel("(Reco - True)/True")
plt.title("2000 Events")
plt.colorbar()  
plt.show()
plt.savefig('dedx_true_reco_diff_4000_events.png')