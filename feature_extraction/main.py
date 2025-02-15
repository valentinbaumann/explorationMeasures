
#### main script for the extraction of movement features and plotting of movement trajectories
#### relies on functions from readData.py, statisticalMeasures.py, visual.py


#%%
# configure paths and general settings

############
#  import libraries and set paths
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from skimage import io
from scipy import spatial as sp

############
# configure image rendering
#pio.renderers.default = 'svg' # plotly directly generates an svg file
pio.renderers.default = 'browser' # plotly output directlys shows in the browser (better for dynamic plotly graphics)

############
# define paths for input paths
#home = os.getcwd() # use a dynamic home directory 
home = 'C:/Users/Valentin/Desktop/KKJP/novelty_lifespan_exploration/analysis_silcton/FeatureExtraction' # driectly defines the home directory 
pathToSourceFiles = os.path.join(home, "feature_extraction")
pathToPosLogsNEMO = os.path.join(home, 'position_data_NEMO')  
pathToFlightLogsNEMO = os.path.join(home, 'flight_data_NEMO')  
pathToPosLogsSilctonExp1 = os.path.join(home, 'position_data_SILCTON_EXP1')  
pathToPosLogsSilctonExp2 = os.path.join(home, 'position_data_SILCTON_EXP2')  
pathToFlightLogsSILCTON = os.path.join(home, 'flight_data_SILCTON')  
pathToPersonData = os.path.join(home, 'person_data') 
pathToObjects = os.path.join(home, "landmark_data")
pathToOutput = os.path.join(home, "output_data")

############
# import custom functions for data loading (readData), feature extraction (statisticalMeasures) and visualization (visual)
os.chdir(pathToSourceFiles)
import readData as readData # import data 
import statisticalMeasure as stm # computation of features
import visual as vis # plotting # visualization fucntions (heatmaps etc.)



#%%
### read data
print("load data")

# CAVE: df_persons and path_data MUST be of same length and indexing, as the trajectories are only indexed by their position within the respective list
    
###############
# get person data

# NEMO
df_persons_NEMO = pd.read_csv(os.path.join(pathToPersonData, 'person_data_NEMO.csv'))
df_persons_NEMO["Subject"] = df_persons_NEMO["Subject"].astype(int).astype(str) # convert to int first to prevent creation of decimals

# SILCTON
df_persons_SILCTON_exp1 = pd.read_csv(os.path.join(pathToPersonData, 'person_data_SILCTON_exp1.csv'))
df_persons_SILCTON_exp2 = pd.read_csv(os.path.join(pathToPersonData, 'person_data_SILCTON_exp2.csv'))
# delete unnecessary columns
df_persons_SILCTON_exp1 = df_persons_SILCTON_exp1[["Subject", "gender", "abs_error"]]
df_persons_SILCTON_exp2 = df_persons_SILCTON_exp2[["Subject", "gender", "abs_error"]]
# add EXP id
df_persons_SILCTON_exp1 [["exp"]] = 1
df_persons_SILCTON_exp2 [["exp"]] = 2
# add both datasets together and create unique subject IDs
df_persons_SILCTON = pd.concat([df_persons_SILCTON_exp1, df_persons_SILCTON_exp2])
df_persons_SILCTON ["Subject"] = df_persons_SILCTON["exp"].astype(int).astype(str) + "_" +  df_persons_SILCTON["Subject"].astype(int).astype(str) 

###############
# get trajectories
# NEMO
path_data_NEMO = readData.loadLogsFromCSV(pathToPosLogsNEMO, sep = ",", header = None)
# SILCTON
path_data_SILCTON_EXP1 = readData.loadLogsFromCSV(pathToPosLogsSilctonExp1, sep = ";", header = None)
path_data_SILCTON_EXP2 = readData.loadLogsFromCSV(pathToPosLogsSilctonExp2, sep = ";", header = None)
path_data_SILCTON = path_data_SILCTON_EXP1 + path_data_SILCTON_EXP2 # add trajecotries from both experiments

###############
# load object data
os.chdir(pathToObjects)
# NEMO
df_obj_green = pd.read_csv('objectsNEMO_green.csv') # read as pandas dataframe
df_obj_pink = pd.read_csv('objectsNEMO_pink.csv') # read as pandas dataframe
obj_green = df_obj_green[['X','Y','Z']].values # convert to numpy array
obj_pink = df_obj_pink[['X','Y','Z']].values # convert to numpy array
# SILCTON
df_obj_SILCTON = pd.read_csv('objectsSILCTON.csv') # read as pandas dataframe
obj_SILCTON= df_obj_SILCTON[['X','Y','Z']].values # convert to numpy array




#%%
###############
# Preprocess trajectories
print("preprocessing trajectories")

#####
# NEMO preprocessing
# unknown samplimg rate, fixed duration
# create a copy of the raw trajectories before trimming / resampling
path_data_NEMO_raw = path_data_NEMO # paths before trimming and resampling
path_data_NEMO_raw_trimmed = readData.trimPathSet(path_data_NEMO) # paths after trimming, but before resampling
# add timestamps
path_data_NEMO = stm.convert3dTo2d(path_data_NEMO) # convert to 2d for resampling procedure
path_data_NEMO = stm.add_timestamps(path_data_NEMO, sessionDuration=150000, samplingInterval="unknown") 
# resample trajectories to 100 ms time intervals (10 Hz sampling rate)
path_data_NEMO = stm.resamplePaths(path_data_NEMO, 100) # resample to a common sampling interval
path_data_NEMO = stm.convert3dTo4d(path_data_NEMO) # add dummy y coordinate so everythjing is in x,y,z,time order
# trim paths (removes initial idle time)
path_data_NEMO = readData.trimPathSet(path_data_NEMO) # trim resampled path
# create a copy in traja format (pandas df) for rediscretization
path_data_NEMO_traj = stm.NumpyToTraja(path_data_NEMO, num_cols=3) 

# create filters for the two NEMO environment types
f_green = np.asarray(df_persons_NEMO.index[df_persons_NEMO['World'] == 'Green'])
f_pink = np.asarray(df_persons_NEMO.index[df_persons_NEMO['World'] == 'Pink'])



#####
# SILCTON preprocessing
# known sampling rate, variable session duration
# drop unnecessary columns present in the original trajectories
path_data_SILCTON = readData.dropColumn(path_data_SILCTON, columns = 3) # drop 4th column
# add coord offset so it matches the landmark coordinates in the objects table
path_data_SILCTON = readData.addCoordOffset(path_data_SILCTON, 708, 688) # correct offset so it mathces the png image
# add timestamps
path_data_SILCTON = stm.add_timestamps(path_data_SILCTON, sessionDuration="variable", samplingInterval = 0.1)
# create a copy of the raw trajectories before trimming 
path_data_SILCTON_raw = path_data_SILCTON
# trim paths (removes initial idle time)
path_data_SILCTON = readData.trimPathSet(path_data_SILCTON) # trim resampled path
# cut all other idle times (due to long waiting times induced by the experiment structure in EXP1)
path_data_SILCTON = stm.deleteIdlePeriodsfromTrajGroup(path_data_SILCTON, duration = 300) # delete all idle periods >= 30s ()
# create a copy in traja format (pandas df) for rediscretization
path_data_SILCTON_traj = stm.NumpyToTraja(path_data_SILCTON, num_cols=3) 


#####
# SILCTON short data (first 8 minutes of movement, idle times are not counted!)
path_data_SILCTONshort = readData.cutPathSet(path_data_SILCTON, 4800)




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
    path_data_NEMO_flights = readData.loadLogsFromCSV(pathToFlightLogsNEMO, sep = ",", header = None)
    path_data_SILCTON_flights = readData.loadLogsFromCSV(pathToFlightLogsSILCTON, sep = ",", header = None)
    print("files loaded from .csv")    

else: # compute from scratch
    path_data_NEMO_flights = stm.getSimplePaths(stm.convert3dTo2d(path_data_NEMO), epsilon=6) # run simplification on 2d paths, since otherwise some steps appear to have values of 0 in 2d space
    path_data_SILCTON_flights = stm.getSimplePaths(stm.convert3dTo2d(path_data_SILCTON), epsilon=14) # run simplification on 2d paths, since otherwise some steps appear to have values of 0 in 2d space

    # save fligth data to csv for quicker access
    # CAVE: make sure that files are saved in the correct order, as otherwise the order of the .txt files may not match the order in the df_persons file!
    readData.saveFlightLogsToCSV(path_data_NEMO_flights, df_persons_NEMO, pathToFlightLogsNEMO) 
    readData.saveFlightLogsToCSV(path_data_SILCTON_flights, df_persons_SILCTON, pathToFlightLogsSILCTON) 

    print("files saved to .csv")




#%%
###############
# compute median step length (used to estimate optimal parameter values for the individual VEs)
print("compute median step length")

# compute median step length 
medianStepLength_NEMO = np.round(np.median(np.concatenate(stm.getHistoryStepLength(path_data_NEMO, "2d", False))), decimals = 2)
medianStepLength_SILCTON = np.round(np.median(np.concatenate(stm.getHistoryStepLength(path_data_SILCTON, "2d", False))), decimals = 2)





#%%
# time parameters 
print("time parameters")

##################
# exploration time (measured in number of datapoints)
# equal to untrimmed/trimmed path length
df_persons_NEMO.loc[:,'TimeRaw'] = stm.getTime(path_data_NEMO_raw) # raw number of datapoints
df_persons_NEMO.loc[:,'TimeRawTrimmed'] = stm.getTime(path_data_NEMO_raw_trimmed) # number of datapoints after trimming, but before resampling
df_persons_NEMO.loc[:,'TimeTrimmed'] = stm.getTime(path_data_NEMO) # number of datapoints after trimming and resampling
df_persons_SILCTON.loc[:,'TimeRaw'] = stm.getTime(path_data_SILCTON_raw)
df_persons_SILCTON.loc[:,'TimeTrimmed'] = stm.getTime(path_data_SILCTON)


##################
# Idle Time as total number of datapoints (mode = "raw"), as percentage (mode = "percent) or as count of idle time periods (via stm.getIdleTimePeriods()))
# on trimmed and resampled paths
df_persons_NEMO.loc[:,'IdleTime'] = stm.getTotalIdleTime(path_data_NEMO, mode = "raw", dim="2d") 
df_persons_SILCTON.loc[:,'IdleTime'] = stm.getTotalIdleTime(path_data_SILCTON, mode = "raw", dim="2d") 



#%%
# distance paramters
print("distance parameters")

##################
# Distance travelled
# relies on stm.computeStepLength()
# dimension can be "2d" for distances on the x-z plane, "3d" for full three dimensional distances or "y" for distances on y axis (y axis distance is used for subject exclusion later on)
# dependent on sampling rate (see Ranacher & Tzavella, 2014)
df_persons_NEMO.loc[:,'PathLength_Y'] = stm.getPathLength(path_data_NEMO_raw, "y") # run this on the original data before resampling 
df_persons_NEMO.loc[:,'PathLength'] = stm.getPathLength(path_data_NEMO, "2d") # here, we take the distance of the resampled paths
df_persons_SILCTON.loc[:,'PathLength'] = stm.getPathLength(path_data_SILCTON, "2d") # here, we take the distance of the resampled paths




#%%
# area parameters
print("area parameters")

optimalBinSize_NEMO = int(30 * medianStepLength_NEMO) # average distance covered in 3 s (rounded down to int)
optimalBinSize_SILCTON = int(30 * medianStepLength_SILCTON) # average distance covered in 3 s (rounded down to int)


##################
# Area covered
# CAVE: depends heavily on "binSize" 

# with binsize estimated from the data
df_persons_NEMO['AreaCovered'] = stm.computeAreaCovered(path_data_NEMO, optimalBinSize_NEMO)
df_persons_SILCTON['AreaCovered'] = stm.computeAreaCovered(path_data_SILCTON, optimalBinSize_SILCTON)


##################
# Roaming Entropy
# relies on stm.computeHistogram()
# depends heavily on bin size 
# may raise "divide by zero" errors during computation

# with binsize estimated from the data
binsCoveredMax_NEMO = stm.defineHist(optimalBinSize_NEMO, path_data_NEMO, world=None)[0][0] * stm.defineHist(optimalBinSize_NEMO, path_data_NEMO, world=None)[0][1]  # get normalization parameter k 
df_persons_NEMO['RoamingEntropy'] = stm.computeRoamingEntropy(path_data_NEMO, optimalBinSize_NEMO, binsCoveredMax_NEMO)

binsCoveredMax_SILCTON = stm.defineHist(optimalBinSize_SILCTON, path_data_SILCTON, world=None)[0][0] * stm.defineHist(optimalBinSize_SILCTON, path_data_SILCTON, world=None)[0][1]  # get normalization parameter k 
df_persons_SILCTON['RoamingEntropy'] = stm.computeRoamingEntropy(path_data_SILCTON, optimalBinSize_SILCTON, binsCoveredMax_SILCTON)




#%%
# minimum convex polygon
print("minimum complex polygon")


df_persons_NEMO.loc[:,'MinimumPolygon'] = stm.getMinimumPolygon(stm.convert3dTo2d(path_data_NEMO))
df_persons_SILCTON.loc[:,'MinimumPolygon'] = stm.getMinimumPolygon(stm.convert3dTo2d(path_data_SILCTON))

# plot convex hull for a single trajectory
# hull = sp.ConvexHull(stm.convert3dTo2d(path_data_NEMO)[0])
# hullPlot = sp.convex_hull_plot_2d(hull)
# plt.show(hullPlot)
##hullPlot.savefig("convexHull.png") 



#%%
# object parameters
# CAVE: may take a while to compute
print("object parameters")

# choose a distance for the circle's radius (visits are counted by entries into a circle around an object)
objDist_NEMO = medianStepLength_NEMO * 20
objDist_SILCTON = medianStepLength_NEMO * 20

##################
# number of objects visited (objects visited at least once)
df_persons_NEMO.loc[f_green,'ObjectsVisited'] = stm.getNumObjVisited([stm.convert3dTo2d(path_data_NEMO)[i] for i in f_green], stm.convert3dTo2d([obj_green])[0], maxDistance=objDist_NEMO, minDuration = 1, minTimeRevisit = 1)
df_persons_NEMO.loc[f_pink,'ObjectsVisited'] = stm.getNumObjVisited([stm.convert3dTo2d(path_data_NEMO)[i] for i in f_pink], stm.convert3dTo2d([obj_pink])[0], maxDistance=objDist_NEMO, minDuration = 1, minTimeRevisit = 1)
df_persons_SILCTON.loc[:,'ObjectsVisited'] = stm.getNumObjVisited(stm.convert3dTo2d(path_data_SILCTON), stm.convert3dTo2d([obj_SILCTON])[0], maxDistance=objDist_SILCTON, minDuration = 0, minTimeRevisit = 0)

##################
# object visits (including revisits)
# use time-resampled data (we use 100ms intervals here)
# minimal visit duration is 0s, minimal revisit time is 0s
df_persons_NEMO.loc[f_green,'ObjectVisits'] = stm.getNumObjVisits([stm.convert3dTo2d(path_data_NEMO)[i] for i in f_green], stm.convert3dTo2d([obj_green])[0], maxDistance=objDist_NEMO, minDuration = 0, minTimeRevisit = 0)
df_persons_NEMO.loc[f_pink,'ObjectVisits'] = stm.getNumObjVisits([stm.convert3dTo2d(path_data_NEMO)[i] for i in f_pink], stm.convert3dTo2d([obj_pink])[0], maxDistance=objDist_NEMO, minDuration = 0, minTimeRevisit = 0)
df_persons_SILCTON.loc[:,'ObjectVisits'] = stm.getNumObjVisits(stm.convert3dTo2d(path_data_SILCTON), stm.convert3dTo2d([obj_SILCTON])[0], maxDistance=objDist_SILCTON, minDuration = 0, minTimeRevisit = 0)

##################
# object revisits
df_persons_NEMO.loc[:,'ObjectRevisits'] = df_persons_NEMO['ObjectVisits'] - df_persons_NEMO['ObjectsVisited'] 
df_persons_SILCTON.loc[:,'ObjectRevisits'] = df_persons_SILCTON['ObjectVisits'] - df_persons_SILCTON['ObjectsVisited'] 

del(objDist_NEMO, objDist_SILCTON)

#%%
##################
# revisiting
# CAVE: takes quite long to compute
print("Revisiting")

# choose a distance for the circle's radius (revisits are counted by entries into a circle)
radiusRevisiting = 14

# Revisiting is computed following the procedure of Gagnon et al. (2016)) 
df_persons_NEMO.loc[:,'Revisiting'] = stm.getRevisiting(stm.convert3dTo2d(path_data_NEMO), radius = radiusRevisiting, method="Gagnon", refractoryPeriod=1)
df_persons_SILCTON.loc[:,'Revisiting'] = stm.getRevisiting(stm.convert3dTo2d(path_data_SILCTON), radius = radiusRevisiting, method="Gagnon", refractoryPeriod=1)
df_persons_SILCTON.loc[:,'Revisiting20'] = stm.getRevisiting(stm.convert3dTo2d(path_data_SILCTON), radius = 20, method="Gagnon", refractoryPeriod=1)

del(radiusRevisiting)



#%%
# sinuosity
# CAVE: may take a while to compute
print("sinuosity")

# rediscretize path for angle and distance calculations, as we use the formula with rediscretization 
# uses the median step length as the new step length
path_data_NEMO_traj_rd = stm.rediscretizePaths(stm.NumpyToTraja(path_data_NEMO, num_cols=3), R = medianStepLength_NEMO)  # rediscretize with a step length of r = 0.48 (corresponds to median step length across all trajectories)
path_data_NEMO_rd = stm.TrajaToNumpy(path_data_NEMO_traj_rd) # convert back to numpy and save as npz
path_data_SILCTON_traj_rd = stm.rediscretizePaths(stm.NumpyToTraja(path_data_SILCTON, num_cols=3), R = medianStepLength_SILCTON)  # rediscretize with a step length of r = 0.48 (corresponds to median step length across all trajectories)
path_data_SILCTON_rd = stm.TrajaToNumpy(path_data_SILCTON_traj_rd) # convert back to numpy and save as npz
        
df_persons_NEMO.loc[:,'Sinuosity'] = stm.getSinuosity(path_data_NEMO_rd, rediscretized=True)
df_persons_SILCTON.loc[:,'Sinuosity'] = stm.getSinuosity(path_data_SILCTON_rd, rediscretized=True)




#%%
##################

# fractal dimension
# CAVE: takes very long to compute
# relies on the traja package
print("fractal dimension")

# get data into traja format
path_data_NEMO_traj = stm.NumpyToTraja(path_data_NEMO, 3)
path_data_SILCTON_traj = stm.NumpyToTraja(path_data_SILCTON, 3)

# get range of step lengths which we will use to calculate fractal dimension
# the range of step lengths represents a logarithmic range of step sizes going from 0.5 x the median step length to 10 x the medain step length
# the number of intervals is specified by numSteps
numSteps = 20
stepSizes_NEMO = stm.CreateLogSequence(0.5 * medianStepLength_NEMO, 10 * medianStepLength_NEMO, numSteps = numSteps) 
stepSizes_SILCTON = stm.CreateLogSequence(0.5 * medianStepLength_SILCTON, 10 * medianStepLength_SILCTON, numSteps = numSteps) 

df_persons_NEMO.loc[:,'FractalDimension'] =  stm.getFractalDimension(path_data_NEMO_traj, stepSizes = stepSizes_NEMO, adjustD = True, meanD = True)
df_persons_SILCTON.loc[:,'FractalDimension'] =  stm.getFractalDimension(path_data_SILCTON_traj, stepSizes = stepSizes_SILCTON, adjustD = True, meanD = True)


del(numSteps, stepSizes_NEMO, stepSizes_SILCTON)





#%%
##################
# get turnarounds
print("turnarounds")

df_persons_NEMO.loc[:,'numTurningPointsFlight160'] = stm.getNumTurningPoints(path_data_NEMO_flights, span = 1, minAngle= 160, angleType="degree")
df_persons_NEMO.loc[:,'numTurningPointsFlight180'] = stm.getNumTurningPoints(path_data_NEMO_flights, span = 1, minAngle= 180, angleType="degree")
df_persons_NEMO.loc[:,'numTurningPointsRaw160'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_NEMO), span = 1, minAngle= 160, angleType="degree")
df_persons_NEMO.loc[:,'numTurningPointsRaw180'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_NEMO), span = 1, minAngle= 180, angleType="degree")

df_persons_SILCTON.loc[:,'numTurningPointsFlight160'] = stm.getNumTurningPoints(path_data_SILCTON_flights, span = 1, minAngle= 160, angleType="degree")
df_persons_SILCTON.loc[:,'numTurningPointsFlight180'] = stm.getNumTurningPoints(path_data_SILCTON_flights, span = 1, minAngle= 180, angleType="degree")
df_persons_SILCTON.loc[:,'numTurningPointsRaw160'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_SILCTON), span = 1, minAngle= 160, angleType="degree")
df_persons_SILCTON.loc[:,'numTurningPointsRaw180'] = stm.getNumTurningPoints(stm.convert3dTo2d(path_data_SILCTON), span = 1, minAngle= 180, angleType="degree" )



#%%
##################
# get efficiency
print("efficiency parameters")

df_persons_NEMO.loc[:,'AreaEfficiency'] = df_persons_NEMO['AreaCovered'] / df_persons_NEMO['PathLength'] 
df_persons_SILCTON.loc[:,'AreaEfficiency'] = df_persons_SILCTON['AreaCovered'] / df_persons_SILCTON['PathLength'] 
df_persons_NEMO.loc[:,'ObjectEfficiency'] = df_persons_NEMO['ObjectsVisited'] / df_persons_NEMO['PathLength'] 
df_persons_SILCTON.loc[:,'ObjectEfficiency'] = df_persons_SILCTON['ObjectsVisited'] / df_persons_SILCTON['PathLength'] 


#%%

#### dataset quality checks
print("data quality checks")

# data quality checks are conducted on trimmed (NEMO and SILCTON), but not resampled data (for NEMO)
# step length variability 
df_persons_NEMO.loc[:,'StepLengthCoeffVar'] = stm.DescribeHistory(stm.getHistoryStepLength(path_data_NEMO_raw_trimmed, "2d", False), "CoeffVar")
df_persons_SILCTON.loc[:,'StepLengthCoeffVar'] = stm.DescribeHistory(stm.getHistoryStepLength(path_data_SILCTON, "2d", False), "CoeffVar")

# get percentage of lag steps (uses median step length as baseline, here we set the cutoff for a "lag step" at 3x the median step length)
df_persons_NEMO.loc[:,'LagStepsPercent'] = stm.getLagStepPercent(stm.getHistoryStepLength(path_data_NEMO_raw_trimmed, "2d"), cutoffMultiplier = 3)
df_persons_SILCTON.loc[:,'LagStepsPercent'] = stm.getLagStepPercent(stm.getHistoryStepLength(path_data_SILCTON, "2d"), cutoffMultiplier = 3)




#%%
##################
# save persons dataframe with all new variables to csv
print("save as csv")

df_persons_NEMO.to_csv (pathToOutput+"/DataExploration_NEMO.csv", index = False, header=True)
df_persons_SILCTON.to_csv (pathToOutput+"/DataExploration_SILCTON.csv", index = False, header=True)





#######


# realtionship of measures to abs pointing error in the SILCTON data
data_performance_SILCTON = df_persons_SILCTON[0:50]

fig = px.scatter(data_performance_SILCTON, x="AreaCovered", y="abs_error", trendline="ols")
fig.show()

fig = px.scatter(data_performance_SILCTON, x="AreaEfficiency", y="abs_error", trendline="ols")
fig.show()
fig = px.scatter(data_performance_SILCTON, x="ObjectEfficiency", y="abs_error", trendline="ols")
fig.show()

fig = px.scatter(data_performance_SILCTON, x="Revisiting", y="abs_error", trendline="ols")
fig.show()

fig = px.scatter(data_performance_SILCTON, x="Sinuosity", y="abs_error", trendline="ols")
fig.show()



#%%
##################
# parameter validity check

# investoagte the effect of different parameter values on the outcome measure (e.g., the effect of binSize on AreaCovered)
# stm.parameterIteration() iterates across a given range of parameters and returns the respective data as a pandas dataframe
print("check parameter validity")

##################
# Area Covered, iterate across "binSize"
parameterValues =[3,7,14,28,56] # choose the respective range of parameters
argument_dict = {'paths': path_data_SILCTON} # define all other arguments used by the function
dataAreaCovered = stm.parameterIteration(parameterValues, stm.computeAreaCovered, argument_dict, "binSize") # result is a pandas dataframe
dataAreaCovered.columns = parameterValues
descriptiveStats = stm.describeDistribution(dataAreaCovered)
descriptiveStats.to_clipboard()
dataAreaCovered.columns = dataAreaCovered.columns.astype(str)

# plot correlation matrix
matrix = dataAreaCovered.corr()
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataAreaCovered)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 


##################
# Roaming Entropy, iterate across "binSize"
parameterValues =[3,7,14,28,56] # choose the respective range of parameters
binsCoveredMax_SILCTON = stm.defineHist(optimalBinSize_SILCTON, path_data_SILCTON, world=None)[0][0] * stm.defineHist(optimalBinSize_SILCTON, path_data_SILCTON, world=None)[0][1]  # get normalization parameter k 
argument_dict = {'paths': path_data_SILCTON, "k": binsCoveredMax_SILCTON} # define all other arguments used by the function
dataRoamingEntropy = stm.parameterIteration(parameterValues, stm.computeRoamingEntropy, argument_dict, "binSize") # result is a pandas dataframe
dataRoamingEntropy.columns = parameterValues
descriptiveStats = stm.describeDistribution(dataRoamingEntropy)
descriptiveStats.to_clipboard()
dataRoamingEntropy.columns = dataRoamingEntropy.columns.astype(str)

# plot correlation matrix
matrix = dataRoamingEntropy.corr()
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataRoamingEntropy)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 




##################
# ObjectsVisited, iterate across "maxDistance"
parameterValues = [5,10,20,40,80] # choose the respective range of parameters
argument_dict = {'paths': stm.convert3dTo2d(path_data_SILCTON), 'objects': stm.convert3dTo2d([obj_SILCTON])[0], 'maxDistance': 10, 'minDuration': 1, 'minTimeRevisit': 1} # define all other arguments used by the function
dataObjectsVisited = stm.parameterIteration(parameterValues, stm.getNumObjVisited, argument_dict, "maxDistance") # result is a pandas dataframe
dataObjectsVisited.columns = parameterValues
descriptiveStats = stm.describeDistribution(dataObjectsVisited)
descriptiveStats.to_clipboard()
dataObjectsVisited.columns = dataObjectsVisited.columns.astype(str)

# plot correlation matrix
matrix = dataObjectsVisited.corr()
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataObjectsVisited)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 


##################
# Object Visits, iterate across "maxDistance"
parameterValues = [5,10,20,40,80] # choose the respective range of parameters
argument_dict = {'paths': stm.convert3dTo2d(path_data_SILCTON), 'objects': stm.convert3dTo2d([obj_SILCTON])[0], 'maxDistance': 10, 'minDuration': 1, 'minTimeRevisit': 1} # define all other arguments used by the function
dataObjectVisits = stm.parameterIteration(parameterValues, stm.getNumObjVisits, argument_dict, "maxDistance") # result is a pandas dataframe
dataObjectVisits.columns = parameterValues
descriptiveStats = stm.describeDistribution(dataObjectVisits)
descriptiveStats.to_clipboard()
dataObjectVisits.columns = dataObjectVisits.columns.astype(str)

# plot correlation matrix
matrix = dataObjectVisits.corr()
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataObjectVisits)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 




##################
# Revisiting, iterate across "radius"
# CAVE: takes very long!
parameterValues = [3,7,14,28,56] # choose the respective range of parameters

argument_dict = {'paths': stm.convert3dTo2d(path_data_SILCTON), 'radius': 10} # define all other arguments used by the function
dataRevisiting = stm.parameterIteration(parameterValues, stm.getRevisiting, argument_dict, "radius") # result is a pandas dataframe
dataRevisiting.columns = parameterValues
descriptiveStats = stm.describeDistribution(dataRevisiting)
descriptiveStats.to_clipboard()
dataRevisiting.columns = dataRevisiting.columns.astype(str)
print(descriptiveStats)
 
# save data to disk to save some time if we want to analyze it again  
os.chdir(pathToOutput)
dataRevisiting.to_csv (pathToOutput+"/DataRevisiting_NEMO.csv", index = False, header=True)
# load data
dataRevisiting = pd.read_csv("DataRevisiting_NEMO.csv", sep=",")

# plot correlation matrix
matrix = dataRevisiting.corr() 
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataRevisiting)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 


##################
# Fractal Dimension, iterate across "stepSizes"
# CAVE: takes very long!
numSteps = 20
stepSizes_SILCTON = stm.CreateLogSequence(0.5 * medianStepLength_SILCTON, 10 * medianStepLength_SILCTON, numSteps = numSteps) 
parameterValues = [stepSizes_SILCTON/4,stepSizes_SILCTON/2,stepSizes_SILCTON,stepSizes_SILCTON*2,stepSizes_SILCTON*4] # choose the respective range of parameters
argument_dict = {'paths': path_data_SILCTON_traj, "adjustD": True, "meanD": True} # define all other arguments used by the function
dataFractalDimension = stm.parameterIteration(parameterValues, stm.getFractalDimension, argument_dict, "stepSizes") # result is a pandas dataframe
colNames = ["1/4", "1/2", "1/1", "2x", "4x"]
dataFractalDimension.columns = colNames
descriptiveStats = stm.describeDistribution(dataFractalDimension)
descriptiveStats.to_clipboard()
dataFractalDimension.columns = dataFractalDimension.columns.astype(str)

# save data to disk to save some time if we want to analyze it again  
os.chdir(pathToOutput)
#dataFractalDimension.to_csv(pathToOutput+"/DataFractalDimension_SILCTON.csv", index = False, header=True)
# load data
dataFractalDimension= pd.read_csv("DataFractalDimension_SILCTON.csv", sep=",")

# plot correlation matrix
matrix = dataFractalDimension.corr() 
sns.heatmap(matrix, cmap="Greens", annot=True)

# plot relationship of all values
os.chdir(pathToOutput)
g = sns.PairGrid(dataFractalDimension)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.map_lower(sns.kdeplot)
fig = g.fig
fig.savefig("out.png") 



##### ALTERNATIVE Plotting (show distributions with the same axis scale for every parameter)
# create long df
colnames = list(dataRevisiting)
dataRevisiting["id"] = dataRevisiting.index
dataRevisitingLong = pd.melt(dataRevisiting, id_vars='id', value_vars=colnames, var_name='parameterValues', value_name='measureValue')

fig = px.histogram(dataRevisitingLong, x = "measureValue", facet_col="parameterValues", nbins=500, facet_col_wrap=3, width=800)
fig.show()





#%%
##################

# plot SILCTON trajectories

# via plotly, plot path and landmarks
my_individuals = [path_data_SILCTON[0]]
fig_individuals = vis.plotPathSet(my_individuals, df_obj_SILCTON)
fig_individuals.update_layout(title = "Subject 1")
fig_individuals.show()

# via plotly, compare two trajectories
my_individuals = [path_data_SILCTON[i] for i in [0, 1]]
fig_individuals = vis.plotPathSet(my_individuals, df_obj_SILCTON)
fig_individuals.update_layout(title = "Subject 1 vs Subject 2")
fig_individuals.show()

# compare raw path vs flight scaled path
rawPath = path_data_SILCTON[0]
flightPath = stm.convert2dTo3d(path_data_SILCTON_flights)[0] # note that plotting at only works for 3d trajectories, so conversipn is needed (dummy axis with all zeros to simulate y axis)
fig_RawVsFlight = vis.plotPathSet([rawPath, flightPath])
fig_RawVsFlight.update_layout(title = "Raw Path vs Path on Flight Scale")
fig_RawVsFlight.show()

# via plotly, plot all trajectories
fig_all = vis.plotPathSet(path_data_SILCTON)
fig_all.update_layout(title = "All Paths")
fig_all.show()


# via plotly, plot objects with object radius
objDist_SILCTON = medianStepLength_NEMO * 20

# plot all object radii
fig_all_object_radii = vis.plotPathSet(path_data_SILCTON[0:1], df_obj_SILCTON)

for landmark in obj_SILCTON:

    leftLimit = landmark[0] - objDist_SILCTON
    print(leftLimit)
    rightLimit = landmark[0] + objDist_SILCTON
    print(rightLimit)
    lowerLimit = landmark[2] - objDist_SILCTON
    upperLimit = landmark[2] + objDist_SILCTON

    fig_all.add_shape(type="circle",
        xref="x", yref="y",
        x0=leftLimit, y0=lowerLimit, x1=rightLimit, y1=upperLimit,
        opacity=0.2,
        fillcolor="blue",
        line_color="blue",
    )
    
fig_all.show()

#############
# via matplotlib: plot objects and trajectories with map
os.chdir(pathToObjects)
SILCTONmap = io.imread("Full_Info_Silcton_Map.png")
os.chdir(pathToOutput)

SILCTON_subset = np.vstack(path_data_SILCTON[0:20]) # take a subset of trajs for better visibility

# via matplotlib, plot map with trajectories and objects
os.chdir(pathToObjects)
im = plt.imread("Full_Info_Silcton_Map.png")
plt.figure(figsize = (20,20))
implot = plt.imshow(im, extent = [0, 1450, 170, 1030], aspect = "equal")
plt.scatter(x = SILCTON_subset[:,0], y = SILCTON_subset[:,2], s = 8)
plt.scatter(x = obj_SILCTON[:,0], y = obj_SILCTON[:,2], s = 100)
plt.show()






#%%

# plot NEMO trajectories

# plot individual path and objects (for first subject)
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[0]], df_obj_green)
fig_path_and_objs.update_layout(title = "Path and Objects")
fig_path_and_objs.show()

# compare multiple individuals (first and second subject) using a filter 
my_individuals = [path_data_NEMO[i] for i in [0, 1]]
fig_individuals = vis.plotPathSet(my_individuals)
fig_individuals.update_layout(title = "Subject 1 vs Subject 2")
fig_individuals.show()

# compare raw path vs flight scaled path
rawPath = path_data_NEMO[0]
flightPath = stm.convert2dTo3d(path_data_NEMO_flights)[0] # note that plotting at only works for 3d trajectories, so conversipn is needed (dummy axis with all zeros to simulate y axis)
fig_RawVsFlight = vis.plotPathSet([rawPath, flightPath])
fig_RawVsFlight.update_layout(title = "Raw Path vs Path on Flight Scale")
fig_RawVsFlight.show()

# visualize all paths per VE
# path_green = [path_data[i] for i in green] # filter result can optionally be stored as an object or inserted directly into the function.
fig_AllPathsGreen = vis.plotPathSet([path_data_NEMO[i] for i in f_green])
fig_AllPathsGreen.update_layout(title = "Green VE")
fig_AllPathsGreen.show()

fig_AllPathsPink = vis.plotPathSet([path_data_NEMO[i] for i in f_pink])
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

# originalPath = path_data_NEMO[477]

# simplePath = rdp.rdp(originalPath, epsilon = 6)
# fig_simplified = vis.plotPathSet([originalPath], df_obj_green)
# fig_simplified.show()

# fig_simplified = vis.plotPathSet([simplePath], df_obj_green)
# fig_simplified.show()


# originalPath = path_data_NEMO[21]

# simplePath = rdp.rdp(originalPath, epsilon = 6)
# fig_simplified = vis.plotPathSet([originalPath], df_obj_green)
# fig_simplified.show()

# fig_simplified = vis.plotPathSet([simplePath], df_obj_green)
# fig_simplified.show()



#%%

# figures for the paper


####### heatmaps 


# create custom boundaries for plots in the article
histBounds = [[-20,150],[-60,90]]
numBins = [18,15]

# Area Covered
hist_green = sum(stm.computeHistBinsEntered([path_data_NEMO[0]], numBins, histBounds))
hist_green_flipped = hist_green.T  
sns.heatmap(np.flip(hist_green_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", cbar_kws=dict(ticks=[]), yticklabels=False, xticklabels=False, clip_on=False)

# Entropy (Probability)
hist_green = stm.computeHistogram([path_data_NEMO[0]], numBins, histBounds)
hist_green_prob = hist_green / np.sum(hist_green)
hist_green_prob_flipped = hist_green_prob.T 
sns.heatmap(np.flip(hist_green_prob_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", yticklabels=False, xticklabels=False, clip_on=False)

# Entropy (Frequency)
hist_green = stm.computeHistogram([path_data_NEMO[0]], numBins, histBounds)
hist_green_freq_flipped = hist_green.T 
sns.heatmap(np.flip(hist_green_freq_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", yticklabels=False, xticklabels=False, clip_on=False)
# no color bar  ticks to make it the same size as the Area Covered histogram
#sns.heatmap(np.flip(hist_green_freq_flipped, axis = 0), linewidths=1, linecolor='grey', cmap="Blues", cbar_kws=dict(ticks=[]), yticklabels=False, xticklabels=False, clip_on=False)



####### trajectories

# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[0]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[0]])#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()


# plot paths and objects together
# draw path
fig = go.Figure(data=go.Scatter(x=path_data_NEMO[0][:,0], y=path_data_NEMO[0][:,2], mode='lines')) 
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
fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_NEMO_flights[0]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()

# simplified trajectory
fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_NEMO_flights[0]]))
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()


# turning angle examples
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[21]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_NEMO_flights[21]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()

# plot individual path and object locations
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[201]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_rdp = vis.plotPathSet(stm.convert2dTo3d([path_data_NEMO_flights[201]]), df_obj_green)
fig_rdp.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_rdp.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_rdp.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_rdp.show()


# Sinuosity examples

data_sin = df_persons_NEMO[["index","Sinuosity","PathLength"]] # get sinuosiyt data only for better data visibility
fig_sin = vis.plotPathSet([path_data_NEMO[189]]) # sinuosity of 0.2
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_NEMO[351]]) # sinuosity of 0.9
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_NEMO[98]]) # sinuosity of 0.9
fig_sin.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_sin.show()

fig_sin = vis.plotPathSet([path_data_NEMO[696]]) # sin of 0.2
fig_sin.update_xaxes(range=[-50, 160], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_sin.update_yaxes(range=[-60, 80], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
fig_sin.show()




##### TURNAROUND examples
# CAVE: some of the trajectories are from NEMO round 2 (not included in the dataset!!) 

# direct plotting
mytraj = path_data_NEMO[0]

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
mytraj = path_data_NEMO[410]
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
fig_path_and_objs = vis.plotPathSet([path_data_NEMO[477]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()

fig_path_and_objs = vis.plotPathSet([path_data_NEMO[218]], df_obj_green)#fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
fig_path_and_objs.update_yaxes(scaleanchor = "x", scaleratio = 1) # enforce a 1:1 aspect ratio
fig_path_and_objs.show()



# flight scale

mytraj = path_data_NEMO_flights[0]
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


mytraj = path_data_NEMO_flights[410]
Angles = stm.computeAngles(mytraj)
Angles160 = np.where(Angles >= 160)[0] # get angles == 180, substract 1 to get original indices in coordinate space
CoordsAngles160 = mytraj[Angles160] # get coordinates of the turning point

myfig = px.line(x = mytraj[:,0], y = mytraj[:,1])
myfig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)",})
myfig.update_xaxes(range=[-65, 130], constrain="domain", showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.update_yaxes(scaleanchor = "x", scaleratio = 1, range=[-55,70], showline=True, linewidth=2, linecolor='Grey', showgrid=True, gridwidth=1, gridcolor='LightGrey', zeroline=True, zerolinewidth=1, zerolinecolor='LightGrey')
myfig.add_trace(go.Scatter(x=CoordsAngles160[:,0], y=CoordsAngles160[:,1], mode='markers', marker=dict(color='LightGrey', size=20, opacity=0.5, line=dict(color='Black',width=2))))
myfig.show()













