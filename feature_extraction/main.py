
#%%

### import libraries and set paths

import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats
import rdp as rdp
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
#pio.renderers.default = 'svg' # plotly will generate svg file
pio.renderers.default = 'browser' # plotly output will show in browser

##############
# define paths for input paths

home = os.getcwd()

pathToSourceFiles = os.path.join(home, "feature_extraction")
pathToLogs = os.path.join(home, 'position_data')  
pathToFlights = os.path.join(home, 'flight_data')  
pathToPersonData = os.path.join(home, 'person_data') 
pathToObjects = os.path.join(home, "feature_extraction")
pathToOutput = os.path.join(home, "output_data")


# import custom functions for data loading (readData), feature extraction (statisticalMeasures) and visualization (visual)
os.chdir(pathToSourceFiles)
import readData as readData # import data from .mat files
import statisticalMeasure as stm # computation of features
import visual as vis # plotting # visualization fucntions (heatmaps etc.)
        
os.chdir(home)



#%%
### read data
print("load data")

# CAVE: df_persons and path_data MUST be of same length and indexing, as the all trajectories are only indexed by their position within the respective list
    
###############
# get person data
df_persons = readData.loadPersonData(pathToPersonData)  # TODO: fix person data import

###############
# load object data
os.chdir(pathToObjects)
df_obj_green = pd.read_csv('objectsGreen.csv') # read as pandas dataframe
df_obj_pink = pd.read_csv('objectsPink.csv') # read as pandas dataframe
obj_green = df_obj_green[['X','Y','Z']].values # convert to numpy array
obj_pink = df_obj_pink[['X','Y','Z']].values # convert to numpy array
 

###############
# load trajectories from csv files
path_data_raw = readData.loadLogsFromCSV(pathToLogs)
path_data_trimmed = readData.trimPathSet(path_data_raw) # trim path

# add timestamps
path_data_raw_2d = stm.convert3dTo2d(path_data_raw) # extract x and z 
path_data_raw_2d_timestamps = stm.add_timestamps(path_data_raw_2d, 150000, "array") # recreate timestamps # TODO: fix timestamp procedure

# resample trajectories to 100 ms time intervals (10 Hz sampling rate)
path_data_raw_resampled = stm.resamplePaths(path_data_raw_2d_timestamps, 100) # resample to a common sampling interval
path_data_raw_resampled = stm.convert3dTo4d(path_data_raw_resampled) # add dummy y coordinate so everythjing is in x,y,z,time order

# trim paths (removes initial idle time)
path_data_resampled = readData.trimPathSet(path_data_raw_resampled) # trim resampled path

# create a copy in traja format (pandas df) for rediscretization
path_data_traj = stm.NumpyToTraja(path_data_resampled, num_cols=3) 

# clean up 
del path_data_raw_2d, path_data_raw_2d_timestamps


#%%
# Path Simplification 
###############
print("path simplification")
# resampling trajectories to flight scale using the ramer-douglas-peucker (rdp) algorithm
# epsilon specifies the maximum acceptable distance of each datapoint to the simplified path form (the higher epsilon, the higher the level of simplification)
# for python implementation also see:  https://rdp.readthedocs.io/en/latest/
# CAVEAT: computationally intensive, may take up to several minutes -> faster: load precomputed files from .csv
# resampled trajectories are in 2d

logs_from_disk = True # TRUE if precomputed files should be loaded, FALSE if data should be computed from scratch

if logs_from_disk == True: # load precomputed files
    path_data_simple_6 = readData.loadLogsFromCSV(pathToFlights)
    print("files loaded from .csv")    

else: # comptue from scratch
    path_data_simple_6 = stm.getSimplePaths(stm.convert3dTo2d(path_data_resampled), epsilon=6) # run simplification on 2d paths, since otherwise some steps appear to have values of 0 in 2d space

    # # save to csv    
    # df_persons['Subject'] = df_persons['Subject'].astype(int).astype(str) # convert to in and then str to remove decimals
    # numSubjects = len(df_persons)

    # for i in range(numSubjects):
    #     trajectory = path_data_simple_6[i]
    #     logName = df_persons["Subject"].values[i]

    #     filePath = pathToFlights + "\\" + logName + "_1" + "_flight" + ".csv"
    #     pd.DataFrame(trajectory).to_csv(filePath, header=False, index=False)

    print("files saved to .csv")





#%%
###############
# create filters for the two environment types
print("create filters")


f_green = np.asarray(df_persons.index[df_persons['World'] == 'Green'])
f_pink = np.asarray(df_persons.index[df_persons['World'] == 'Pink'])




#%%
# time parameters 
print("time parameters")

##################
# exploration time (measured in number of datapoints)
# equal to untrimmed/trimmed path length
df_persons.loc[:,'TimeRaw'] = stm.computeTime(path_data_raw)
#df_persons.loc[:,'TimeTrimmed'] = stm.computeTime(path_data)
# optional: delete untrimmed path data to save some memory
del (path_data_raw) 

##################
# Idle Time as Percentage of total number of datapoints or as count of idle time periods
# relies on stm.getIdleTimePercent() and stm.getIdleTimePeriods()
df_persons.loc[:,'IdleSteps2d'] = stm.getIdleTime(path_data_raw_resampled, mode = "raw", dim="2d") # we compute this on the untrimmed data




#%%
# distance paramters
print("distance parameters")

##################
# Distance travelled
# relies on stm.computeStepLength()
# dimension can be "2d" for distances on the x-z plane, "3d" for full three dimensional distances or "y" for distances on y axis (y axis distance is used for subject exclusion later on)
# dependent on sampling rate (see Ranacher & Tzavella, 2014)
df_persons.loc[:,'DistanceTravelledY'] = stm.computeLength(path_data_trimmed, "y") # run this on the original data before resampling 
df_persons.loc[:,'DistanceTravelled2d_100ms'] = stm.computeLength(path_data_resampled, "2d") # here, we take the distance of the resampled paths




#%%
# area parameters
print("area parameters")


# get optimal bin size
medianStepLength = np.median(np.concatenate(stm.getHistoryStepLength(path_data_resampled, "2d", False))) # median step length across all trajectories (1 step = 100 ms). uses non zero steps only
optimalBinSize = int(30 * medianStepLength) # average distance covered in 3 s (rounded down to int)


# # get histogram sizes 
# stm.defineHist(optimalBinSize, "pink")[0]
# stm.defineHist(optimalBinSize, "green")[0]

# # check if environment sizes are comparable
# # we have a few outliers in green, but overall sizes are comparable
# np.median(df_persons.loc[f_pink,'AreaCoveredRaw_14'])
# np.median(df_persons.loc[f_green,'AreaCoveredRaw_14'])
# px.histogram(df_persons.loc[f_green,'AreaCoveredRaw_14'], range_x=[0,100]) 
# px.histogram(df_persons.loc[f_pink,'AreaCoveredRaw_14'], range_x=[0,100])


##################
# Area covered
# depends heavily on bin size 
# relies on stm.computeHistogram()

# with binsize estimated from the data
df_persons['AreaCoveredRaw_14'] = stm.computeAreaCovered(path_data_resampled, stm.defineHist(optimalBinSize, path_data_resampled, world=None)[0], stm.defineHist(optimalBinSize, path_data_resampled, world=None)[1])


##################
# Roaming Entropy
# relies on stm.computeHistogram()
# depends heavily on bin size 
# may raise "divide by zero" errors during computation

# with binsize estimated from the data
binsCoveredMax = stm.defineHist(optimalBinSize, path_data_resampled, world=None)[0][0] * stm.defineHist(optimalBinSize, path_data_resampled, world=None)[0][1]  # get normalization parameter k 
df_persons['RoamingEntropy_14'] = stm.computeRoamingEntropy(path_data_resampled, stm.defineHist(optimalBinSize, path_data_resampled, world=None)[0], stm.defineHist(optimalBinSize, path_data_resampled, world=None)[1], binsCoveredMax)



#%%
# object parameters
# CAVE: may take a while to compute
print("object parameters")

# we use 2d distances here

##################
# object visits 
df_persons.loc[f_green,'ObjectsVisited10'] = stm.getNumObjVisited([stm.convert3dTo2d(path_data_resampled)[i] for i in f_green], stm.convert3dTo2d([obj_green])[0], maxDistance=10, minDuration = 1, minTimeRevisit = 1)
df_persons.loc[f_pink,'ObjectsVisited10'] = stm.getNumObjVisited([stm.convert3dTo2d(path_data_resampled)[i] for i in f_pink], stm.convert3dTo2d([obj_pink])[0], maxDistance=10, minDuration = 1, minTimeRevisit = 1)

test = [stm.convert3dTo2d(path_data_resampled)[i] for i in f_green]

##################
# object revisits
# use time-resampled data (we use 100ms intervals here)
# minimal visit duration is 0s, minimal revisit time is 0s

df_persons.loc[f_green,'ObjectVisits10'] = stm.getNumObjVisits([stm.convert3dTo2d(path_data_resampled)[i] for i in f_green], stm.convert3dTo2d([obj_green])[0], maxDistance=10, minDuration = 0, minTimeRevisit = 0)
df_persons.loc[f_pink,'ObjectVisits10'] = stm.getNumObjVisits([stm.convert3dTo2d(path_data_resampled)[i] for i in f_pink], stm.convert3dTo2d([obj_pink])[0], maxDistance=10, minDuration = 0, minTimeRevisit = 0)



#%%
##################
# revisiting
# CAVE: may take a while to compute
print("Revisiting")

# get optimal radius
medianStepLength = np.median(np.concatenate(stm.getHistoryStepLength(path_data_resampled, "2d", False))) # median step length across all trajectories (1 step = 100 ms). uses non zero steps only
optimalRadius= int(30 * medianStepLength) # average distance covered in 3 s (rounded down to int)

df_persons.loc[:,'RevisitingGagnon14'] = stm.getRevisiting(stm.convert3dTo2d(path_data_resampled), radius = optimalRadius, method="Gagnon", refractoryPeriod=1)





#%%

#### dataset quality checks
print("data quality checks")

# step length variability (on trimmed but not resampled data)
df_persons.loc[:,'StepLengthCoeffVar'] = stm.DescribeHistory(stm.getHistoryStepLength(path_data_trimmed, "2d", False), "CoeffVar")

# get percentage of lag steps (cutoff at 3x the median step length)
df_persons.loc[:,'LagStepsPercent3xMedian'] = stm.getLagStepPercent(stm.getHistoryStepLength(path_data_resampled, "2d"), cutoffMultiplier = 3)
df_persons.loc[:,'LagStepsPercent4xMedian'] = stm.getLagStepPercent(stm.getHistoryStepLength(path_data_resampled, "2d"), cutoffMultiplier = 4)
df_persons.loc[:,'LagStepsPercent5xMedian'] = stm.getLagStepPercent(stm.getHistoryStepLength(path_data_resampled, "2d"), cutoffMultiplier = 5)

# get average data loss through trimming (as number of datapoints lost)
trimDiffs = []
for i in range(len(path_data_raw_resampled)):
    diff = len(path_data_raw_resampled[i]) - len(path_data_resampled[i])
    trimDiffs.append(diff)
df_persons.loc[:,'trimLoss'] = np.asarray(trimDiffs)




#%%
# sinuosity
# CAVE: may take a while to compute
print("sinuosity")

medianStepLength = np.median(np.concatenate(stm.getHistoryStepLength(path_data_resampled, "2d", False)))
medianStepLength = np.round(medianStepLength, decimals= 2)

# rediscretize path for angle and distance calculations
path_data_resampled_traj_rd = stm.rediscretizePaths(stm.NumpyToTraja(path_data_resampled, num_cols=3), R = medianStepLength)  # rediscretize with a step length of r = 0.48 (corresponds to median step length across all trajectories)
path_data_resampled_rd = stm.TrajaToNumpy(path_data_resampled_traj_rd) # convert back to numpy and save as npz
    

df_persons.loc[:,'SinuosityRediscretized'] = stm.getSinuosity(path_data_resampled_rd, rediscretized=True)




#%%
##################

# fractal dimension
# CAVE: may take a while to compute
print("fractal dimension")

# resample data to 100ms and get data into traja format
path_data_resampled_traj = stm.NumpyToTraja(path_data_resampled, 3)

# get range of step lengths which we will use to calculate fractal dimension
# we use the mean step length across participants as a reference point here
# stepSizes in Paulus et al. (1991) unknown / "steps" are based on a fixed distance due to measurement technique?
meanStepLength = np.nanmean(stm.DescribeHistory(stm.getHistoryStepLength(path_data_resampled, "2d", False), "mean"))
startSeq = 0.5 * meanStepLength
stopSeq = 10 * meanStepLength
numSteps = 20
stepSizes = stm.CreateLogSequence(startSeq, stopSeq, numSteps = numSteps)

df_persons.loc[:,'FractalDimension'] =  stm.getFractalDimension(path_data_resampled_traj, stepSizes = stepSizes, adjustD = True, meanD = True)





#%%
##################
# get turnarounds
print("turnarounds")

df_persons.loc[:,'numTurningPointsFlight150'] = stm.getNumTurningPoints(path_data_simple_6, 150)
df_persons.loc[:,'numTurningPointsFlight160'] = stm.getNumTurningPoints(path_data_simple_6, 160)
df_persons.loc[:,'numTurningPointsFlight170'] = stm.getNumTurningPoints(path_data_simple_6, 170)
df_persons.loc[:,'numTurningPointsFlight180'] = stm.getNumTurningPoints(path_data_simple_6, 180)


df_persons.loc[:,'numTurningPointsRaw150'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_resampled), 150)
df_persons.loc[:,'numTurningPointsRaw160'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_resampled), 160)
df_persons.loc[:,'numTurningPointsRaw170'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_resampled), 170)
df_persons.loc[:,'numTurningPointsRaw180'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_resampled), 180)




#%%
##################
# save persons dataframe with all new variables to csv
print("save as csv")

df_persons.to_csv (pathToOutput+"/DataExploration.csv", index = False, header=True)









#%%

# plot trajectories


# plot individual path and objects (for first subject)
fig_path_and_objs = vis.plotPathSet([path_data_resampled[0]], df_obj_green)
fig_path_and_objs.update_layout(title = "Path and Objects")
fig_path_and_objs.show()


# compare multiple individuals (first and second subject) using a filter 
my_individuals = [path_data_resampled[i] for i in [0, 1]]
fig_individuals = vis.plotPathSet(my_individuals)
fig_individuals.update_layout(title = "Subject 1 vs Subject 2")
fig_individuals.show()

# compare raw path vs flight scaled path
rawPath = path_data_resampled[0]
flightPath = stm.convert2dTo3d(path_data_simple_6)[0] # note that plotting at only works for 3d trajectories, so conversipn is needed (dummy axis with all zeros to simulate y axis)
fig_RawVsFlight = vis.plotPathSet([rawPath, flightPath])
fig_RawVsFlight.update_layout(title = "Raw Path vs Path on Flight Scale")
fig_RawVsFlight.show()

# visualize all paths per VE
# path_green = [path_data[i] for i in green] # filter result can optionally be stored as an object or inserted directly into the function.
fig_AllPathsGreen = vis.plotPathSet([path_data_resampled[i] for i in f_green])
fig_AllPathsGreen.update_layout(title = "Green VE")
fig_AllPathsGreen.show()

fig_AllPathsPink = vis.plotPathSet([path_data_resampled[i] for i in f_pink])
fig_AllPathsPink.update_layout(title = "Pink VE", plot_bgcolor='rgba(0,0,0,0)')
fig_AllPathsPink.show()



#%%

# plot objects
    
fig_objects_green = vis.plotObjectLocations(df_obj_green)
fig_objects_green.update_layout(title = "Green objects")
fig_objects_green.show()
    
fig_objects_pink = vis.plotObjectLocations(df_obj_pink) 
fig_objects_pink.update_layout(title = "Pink objects")
fig_objects_pink.show()   
    


#%%
# show that  turnaround calculation via RDP is better than Farran et al. (2013)

# originalPath = path_data_resampled[477]

# simplePath = rdp.rdp(originalPath, epsilon = 6)
# fig_simplified = vis.plotPathSet([originalPath], df_obj_green)
# fig_simplified.show()

# fig_simplified = vis.plotPathSet([simplePath], df_obj_green)
# fig_simplified.show()


# originalPath = path_data_resampled[21]

# simplePath = rdp.rdp(originalPath, epsilon = 6)
# fig_simplified = vis.plotPathSet([originalPath], df_obj_green)
# fig_simplified.show()

# fig_simplified = vis.plotPathSet([simplePath], df_obj_green)
# fig_simplified.show()



#%%

# figures for the paper


####### heatmaps 
import seaborn as sns

# create custom boundaries for plots in the article
histBounds = [[-20,150],[-60,90]]
numBins = [18,15]


# Area Covered
hist_green = sum(stm.computeHistBinsEntered([path_data_resampled[0]], numBins, histBounds))
hist_green_flipped = hist_green.T  
sns.heatmap(np.flip(hist_green_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", cbar_kws=dict(ticks=[]), yticklabels=False, xticklabels=False, clip_on=False)

# Entropy (Probability)
hist_green = stm.computeHistogram([path_data_resampled[0]], numBins, histBounds)
hist_green_prob = hist_green / np.sum(hist_green)
hist_green_prob_flipped = hist_green_prob.T 
sns.heatmap(np.flip(hist_green_prob_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", yticklabels=False, xticklabels=False, clip_on=False)

# Entropy (Frequency)
hist_green = stm.computeHistogram([path_data_resampled[0]], numBins, histBounds)
hist_green_freq_flipped = hist_green.T 
sns.heatmap(np.flip(hist_green_freq_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", yticklabels=False, xticklabels=False, clip_on=False)
# no color bar  ticks to make it the same size as the Area Covered histogram
sns.heatmap(np.flip(hist_green_freq_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", cbar_kws=dict(ticks=[]), yticklabels=False, xticklabels=False, clip_on=False)



####### trajectories

# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet([path_data_resampled[0]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()


mypath = [path_data_resampled[0]]
# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet(mypath)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()



# plot paths and objects together
# draw path
fig = go.Figure(data=go.Scatter(x=path_data_resampled[0][:,0], y=path_data_resampled[0][:,2], mode='lines')) 
# add object locations
fig.add_trace(go.Scatter(
                x=df_obj_green['X'],
                y=df_obj_green['Z'],
                mode='markers',
                marker=dict(
                    color='LightSkyBlue',
                    size=50,
                    opacity=0.5,
                    line=dict(
                        color='MediumPurple',
                        width=2
                    )
                ),
                #text=myobjects['Landmark'],
                #textposition='top right',
                #mode='markers+text',
                #name='Objects'
                ))
fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig.show()
 #fig.write_image("fig1.svg") # export to svg




# simplified trajectory
fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_simple_6[0]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()




# simplified trajectory
fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_simple_6[0]]))
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()


# turning angle examples
fig_path_and_objs = vis.plotPathSet([path_data_resampled[21]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_simple_6[21]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()

# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet([path_data_resampled[201]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_simple_6[201]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()


# Sinuosity examples

data_sin = df_persons[["index","SinuosityRediscretized","DistanceTravelled2d"]]
fig_sin = vis.plotPathSet([path_data_resampled[189]]) # sinuosity of 0.2
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_resampled[351]]) # sinuosity of 0.9
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_resampled[98]]) # sinuosity of 0.9
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_resampled[696]]) # sin of 0.2
fig_sin.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_sin.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_sin.show()




##### TURNAROUND examples

# direct plotting
mytraj = path_data_resampled[0]

Angles = stm.computeAngles(stm.convert3dTo2d([mytraj])[0])
Angles180 = np.where(Angles == 180)[0] # get angles == 180, substract 1 to get original indices in coordinate space
CoordsAngles180 = mytraj[Angles180] # get coordinates of the turning point

myfig = px.line(x = mytraj[:,0], y = mytraj[:,2])
myfig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
myfig.update_xaxes(range=[-45, 150], constrain="domain", showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.update_yaxes(scaleanchor = "x", scaleratio = 1, range=[-45,80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.add_trace(go.Scatter(x=CoordsAngles180[:,0], y=CoordsAngles180[:,2], mode='markers', marker=dict(color='LightGrey', size=20, opacity=0.5, line=dict(color='Black',width=2))))
myfig.update_layout(yaxis = dict(tickfont = dict(size=20)), xaxis = dict(tickfont = dict(size=20)))
myfig.show()

# direct plotting
mytraj = path_data_resampled[410]
Angles = stm.computeAngles(stm.convert3dTo2d([mytraj])[0])
Angles180 = np.where(Angles == 180)[0] # get angles == 180, substract 1 to get original indices in coordinate space
CoordsAngles180 = mytraj[Angles180] # get coordinates of the turning point

myfig = px.line(x = mytraj[:,0], y = mytraj[:,2])
myfig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
myfig.update_xaxes(range=[-65, 130], constrain="domain", showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.update_yaxes(scaleanchor = "x", scaleratio = 1, range=[-55,70], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.add_trace(go.Scatter(x=CoordsAngles180[:,0], y=CoordsAngles180[:,2], mode='markers', marker=dict(color='LightGrey', size=20, opacity=0.5, line=dict(color='Black',width=2))))
myfig.show()


# Trunaround Examples (OLD)
fig_path_and_objs = vis.plotPathSet([path_data_resampled[477]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_path_and_objs = vis.plotPathSet([path_data_resampled[218]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()



# flight scale

mytraj = path_data_simple_6[0]
Angles = stm.computeAngles(mytraj)
Angles160 = np.where(Angles >= 160)[0] # get angles == 180, substract 1 to get original indices in coordinate space
CoordsAngles160 = mytraj[Angles160] # get coordinates of the turning point

myfig = px.line(x = mytraj[:,0], y = mytraj[:,1])
myfig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
myfig.update_xaxes(range=[-50, 150], constrain="domain", showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.update_yaxes(scaleanchor = "x", scaleratio = 1, range=[-60,80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.add_trace(go.Scatter(x=CoordsAngles160[:,0], y=CoordsAngles160[:,1], mode='markers', marker=dict(color='LightGrey', size=20, opacity=0.5, line=dict(color='Black',width=2))))
myfig.update_layout(yaxis = dict(tickfont = dict(size=20)), xaxis = dict(tickfont = dict(size=20)))
myfig.show()


mytraj = path_data_simple_6[410]
Angles = stm.computeAngles(mytraj)
Angles160 = np.where(Angles >= 160)[0] # get angles == 180, substract 1 to get original indices in coordinate space
CoordsAngles160 = mytraj[Angles160] # get coordinates of the turning point

myfig = px.line(x = mytraj[:,0], y = mytraj[:,1])
myfig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
myfig.update_xaxes(range=[-65, 130], constrain="domain", showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.update_yaxes(scaleanchor = "x", scaleratio = 1, range=[-55,70], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.add_trace(go.Scatter(x=CoordsAngles160[:,0], y=CoordsAngles160[:,1], mode='markers', marker=dict(color='LightGrey', size=20, opacity=0.5, line=dict(color='Black',width=2))))
myfig.show()








