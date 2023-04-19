'''
Shower dEdx seperation with 1 voxel scan. Finding median and mean methods.
'''

import os, sys
SOFTWARE_DIR = '%s/lartpc_mlreco3d' % os.environ.get('HOME')#%s/lartpc_mlreco3d' % os.environ.get('HOME') #'/sdf/group/neutrino/drielsma/me/lartpc_mlreco3d'
DATA_DIR = '/sdf/home/d/dcarber/DATADIR'
# Set software directory
sys.path.append(SOFTWARE_DIR)
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import math
from math import ceil, floor
seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_context('talk')
print(SOFTWARE_DIR)
from pprint import pprint
#pprint(particles)
#pprint(true_particles)
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

cfg = yaml.load(open('%s/inference_08022022.cfg' % DATA_DIR, 'r').read().replace('DATA_DIR', DATA_DIR),Loader=yaml.Loader)
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
radius = 13 # in voxels
thickness = 3
pca = PCA(n_components=2)
def compute_shower_dqdx(matched_particles,matched_interaction, r=10, min_segment_size=3):
    '''
    Inputs:
        -matched_particles(list of matched particles)
        -matched_interaction(list of matched interactions)
    Returns:
        -out: list of computed dQ/dx for each particle
        -out_electron: list of computed dQ/dx for true electrons
        -out_photon: list of computed dQ/dx for true photons
        -out_true: list of true dQ/dx for each particle
        -out_total: list of true total energy for each particle
    '''

    out= []
    out_electron = []
    out_photon = []
    out_true = []
    out_total =[]
    out_true_photon = []
    out_true_electron = []
    depo1 = []
    depo =[]
    
    particles = []
    true_particles = []
    #print(matched_interaction)
    
    interaction_vertexs = matched_interaction[0][0].vertex
    interaction_vertex = np.array(interaction_vertexs)
    #print(interaction_vertex, " and ", matched_interaction[0][0].vertex)
    point = []
    
    min_x, min_y, min_z = data['meta'][entry][0:3]
    max_x, max_y, max_z = data['meta'][entry][3:6]
    size_voxel_x, size_voxel_y, size_voxel_z = data['meta'][entry][6:9]
    tdrift = 0
    for m in matched_particles: #Seperates particles and their matched true particle
        particles.append(m[0])
        true_particles.append(m[1])
    for i in range(len(matched_particles)): #Loops over all particles in the matched particle list
        #Makes sure the particles are what we want
        if true_particles[i] == None: #Gets rid of any particle that has no matched true particle
            continue
        if(particles[i].pid > 1): #Selects either photon or electron particles
            continue
        if(particles[i].semantic_type > 1):#Selects fragment showers
            continue
        assert particles[i].is_primary #Checks if the particle is a primary particle
        if (particles[i].startpoint < 0).any():
            continue
            
        # Find the startpoint    
        point_dist =[]
        for points in particles[i].points: #Finds distance from points in the shower and the vertex of the interaction
            point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
        closest_point = np.argmin(point_dist)#Finds the point with the minimum distance 
        #If the PPN startpoint is within a certain distance from the closest point to vertex it will choose the PPN start point
        if cdist(particles[i].points[closest_point].reshape(1,-1),particles[i].startpoint.reshape(1,-1)) >= 4: 
            for points in particles[i].points:
                #print(interaction_vertex)
                point_dist.append(cdist(interaction_vertex.reshape(1,-1),points.reshape(1,-1)))
            closest_point = np.argmin(point_dist)
            ppn_prediction = particles[i].points[closest_point]
        else:
            ppn_prediction = particles[i].startpoint
            
        #ppn_prediction = particles[i].startpoint
        true_startpoint = true_particles[i].startpoint
        true_dist = cdist(true_particles[i].points, true_startpoint.reshape(1,-1))
        #true_mask = true_dist.squeeze() < r
       
        dist = cdist(particles[i].points, ppn_prediction.reshape(1,-1))#Distance of each point in particle from the start point
        max_dist = np.where(dist == max(dist))
        #print("Max dist point ",max_dist[0],", Max dist ", dist[max_dist[0]],", max(dist) ", max(dist) )
        max_point = particles[i].points[max_dist[0]]
        #mask = dist.squeeze() < r #Are the points' distance from start point within selected radius
        mask = []
        true_mask = []
        for td in true_dist:
            if td < r  and td > 1.5:
                true_mask.append(True)
            else:
                true_mask.append(False)
        for d in dist:
            vertex_dist = cdist(ppn_prediction.reshape(1,-1), interaction_vertex.reshape(1,-1))
            if vertex_dist > 1.5:
                mask = dist.squeeze() < r
                break
            if d < r and d > 1.5:
                mask.append(True)
                #print(particles[i].depositions[np.where(dist ==d)[0][0]])
                #depo1.append(particles[i].depositions[np.where(dist ==d)[0][0]])
            else:
                mask.append(False)
        selected_points = particles[i].points[mask]#Grabs points that are within the radius
        true_selected_points = true_particles[i].points[true_mask]
        if selected_points.shape[0] < 2:# Makes sure there are enough points
            continue
        if true_selected_points.shape[0] < 2:# Makes sure there are enough points
            continue
        #Finds the drift time to the corresponding anode to calculate the calibration from ADC to MeV
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
        #Grabs dq and dx information
        proj = pca.fit_transform(selected_points)
        selected_depositions = particles[i].depositions[mask]*np.exp(tdrift/3000)
        dqdx_list = []
        dedx_list = []
        for r in range(math.floor(proj[:,0].min()),math.ceil(proj[:,0].max()),1):#math.floor(proj[:,0].min()
            dqdx = 0
            for sel in range(len(selected_points)):
                #print(proj[sel,0],type(r))
                if proj[sel,0] >= r and proj[sel,0] <= r+1:
                    dqdx = dqdx+((selected_depositions[sel]*np.exp(tdrift/3000))/.3)
                    #print("hi",dqdx, selected_depositions[sel])
            dqdx_list.append(dqdx)
        #print("List: ",dqdx_list)
        mean_dqdx = np.mean(dqdx_list)
        rms_dqdx = np.sqrt(mean_dqdx**2)
        for de in dqdx_list:
            if de <= mean_dqdx+rms_dqdx:
                dedx_list.append(de)
        median_dqdx = np.median(dedx_list)
        mean_dqdx = np.mean(dedx_list)
        #print("mean: ",mean_dqdx," median: ",median_dqdx)
        #print(proj[:,0], proj[:,0].max(), proj)
        dx = (proj[:,0].max() - proj[:,0].min())
        if dx < min_segment_size:
            continue
        dx = dx*.3
        #print(tdrift)
        #dq = np.sum(particles[i].depositions[mask])#*(23.6*10**(-6))*(86.83)*np.exp(tdrift/3000)*0.66**(-1)
        #dq = np.median(particles[i].depositions[mask])*np.exp(tdrift/3000)#*np.sum(mask)
        #print(np.sum(mask))
        #dedx = (np.exp(((dq/dx)*(23.6*10**(-6))*86.83*.212)/(1.383*.5))-.93)/(.212/(1.383*.5))
        dedx = (np.exp(((median_dqdx)*(23.6*10**(-6))*86.83*.212)/(1.383*.5))-.93)/(.212/(1.383*.5))
        #dq = dq*(23.6*10**(-6))*(86.83)*0.66**(-1)
        dq_true = np.sum(true_particles[i].depositions_MeV[true_mask])
        #print(dq)
        #dedx = (np.exp((dq/dx)*.212/(1.383*.5))-.93)/(.212/(1.383*.5))
        
        total_energy = np.sum(true_particles[i].depositions_MeV)
        #diff = (dq/dx - dq_true/dx)/(dq_true/dx)
            
        #if diff > -.01 and diff <.01 :
            #print("Hello", particles[i].id)
        #print(dedx)
        out_total.append(dq_true)
        out.append(dedx)
        out_true.append(dq_true/dx)
            #print(particles[i].depositions[mask])
            #depo.append(particles[i].depositions[mask])
            #start.append(cdist(interaction_vertex.reshape(1,-1),ppn_prediction.reshape(1,-1)))
        #ppn.append(distance)
        if true_particles[i].pid == 0:
            out_photon.append(dedx)
            out_true_photon.append(dq_true/dx)
        if true_particles[i].pid == 1:
            out_electron.append(dedx)
            out_true_electron.append(dq_true/dx)
     
    return out_photon, out_electron, out, out_true, out_true_photon, out_true_electron
    #return out_photon, out_electron, out, out_true, out_total, out_true_electron, out_true_photon, start, ppn
iterations = 100
collect_dqdx_electrons = []
collect_dqdx_photons = []
collect_dqdx = []
collect_dqdx_true = []
collect_dqdx_total = []
collect_dqdx_true_electrons = []
collect_dqdx_true_photons = []
depo1 = [] 
depo = []
sel_e = 0

for iteration in range(iterations):
    data, result = hs.trainer.forward(dataset)
    evaluator = FullChainEvaluator(data, result, cfg, deghosting=True)
    print(iteration)
    for entry, index in enumerate(evaluator.index):
        #print("Batch ID: {}, Index: {}".format(entry, index))
        matched_particles = evaluator.match_particles(entry, only_primaries = True,mode = 'pred_to_true')
        matched_interaction = evaluator.match_interactions(entry)
        if matched_interaction == []:
            continue
        #print(matched_particles)
        #print(matched_interaction)
        dqdx = compute_shower_dqdx(matched_particles,matched_interaction, r=radius, min_segment_size=min_segment_size)
        '''collect_dqdx_photons.extend(dqdx[0])
        collect_dqdx_electrons.extend(dqdx[1])
        collect_dqdx.extend(dqdx[2])
        collect_dqdx_true.extend(dqdx[3])
        collect_dqdx_total.extend(dqdx[4])
        collect_dqdx_true_electrons.extend(dqdx[5])
        collect_dqdx_true_photons.extend(dqdx[6])
        start.extend(dqdx[7])
        ppn.extend(dqdx[8])
        
collect_dqdx_electrons = np.array(collect_dqdx_electrons)
collect_dqdx_photons = np.array(collect_dqdx_photons)
collect_dqdx = np.array(collect_dqdx)
collect_dqdx_true = np.array(collect_dqdx_true)
collect_dqdx_total = np.array(collect_dqdx_total)
collect_dqdx_true_photons = np.array(collect_dqdx_true_photons)
collect_dqdx_true_electrons = np.array(collect_dqdx_true_electrons)
start = np.array(start)
ppn = np.array(ppn)'''
        collect_dqdx_photons.extend(dqdx[0])
        collect_dqdx_electrons.extend(dqdx[1])
        collect_dqdx.extend(dqdx[2])
        collect_dqdx_true.extend(dqdx[3])
        depo1.extend(dqdx[4])
        depo.extend(dqdx[5])
collect_dqdx_electrons = np.array(collect_dqdx_electrons)
collect_dqdx_photons = np.array(collect_dqdx_photons)
collect_dqdx = np.array(collect_dqdx)
collect_dqdx_true = np.array(collect_dqdx_true)
depo1 = np.array(depo1)
depo = np.array(depo)
print("Median for electrons: ",np.median(collect_dqdx_electrons)," Median for photons: ", np.median(collect_dqdx_photons))
print("Mean for electrons: " , collect_dqdx_electrons.mean()," Mean for photons: ", collect_dqdx_photons.mean())
print("Total electrons ",len(collect_dqdx_electrons),", Total photons ", len(collect_dqdx_photons))
for e in collect_dqdx_electrons:
    if e < 3:
        sel_e += 1
purity = sel_e/len(collect_dqdx_electrons)
efficiency = sel_e/(len(collect_dqdx_electrons)+len(collect_dqdx_photons))
print("Purity: ", purity, ", Efficiency: ", efficiency)

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
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-')
fig = plt.subplots(figsize =(10, 6))
# Creating plot
#x_bins = np.linspace(0, 20, 1)
#y_bins = np.linspace(0, 20,1)

#plot2 = plt.hist2d(x,y,range = [[0, 1.25], [0, 1.25]], bins = [20,20])
plot1 = plt.hist2d(collect_dqdx, collect_dqdx_true,range = [[0, 1], [0, 1]], bins = [100,100])
#plot2 = plt.hist2d(x,y,range = [[0, 1.25], [0, 1.25]], bins = [20,20])
abline(1,0)
plt.title("True Shower Start vs. Reco Shower Start")
plt.xlabel("Reco dE/dx [MeV/cm]")
plt.ylabel("Truth dE/dx [MeV/cm]")
plt.colorbar()  
# show plot
plt.show()
diff = []
std =[]
x_bins = 100
num =0
diff = (np.subtract(collect_dqdx,collect_dqdx_true))/collect_dqdx_true
fig = plt.subplots(figsize =(10, 7))
#fig = plt.figure()
#gs = fig.add_gridspec(2, hspace=0)
#axs = gs.subplots(sharex=True, sharey=True)
#plot2 = plt.hist2d(x,y,range = [[0, 1.25], [0, 1.25]], bins = [20,20])
for i in range(x_bins):
    values = []
    for t in range(len(collect_dqdx_true)):
        num =collect_dqdx_true[t]
        if num>=i*10 and num<(i+1)*10:
            values.append(diff[t])
    std.append(np.std(values))
    #print(std)
#plot2 = plt.hist(std,bins = x_bins, range = [0,x_bins*10])
plot1 = plt.hist2d(collect_dqdx_true, diff,range = [[0, 10], [-1, 1]], bins = [x_bins,100])

     
    
#plt.title("True Shower Start vs. Reco Shower Start")
plt.xlabel("True dE/dx [MeV/cm]")
plt.ylabel("(Reco - True)/True")
plt.title("2000 Events")
plt.colorbar()  
# show plot
plt.show()
plt.savefig('dedx_true_reco_diff_4000_events.png')