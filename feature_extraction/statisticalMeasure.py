#### functions to compute all exploration measures from the individual trajectories

# note that functions containing the keyword "compute" target a single trajectory, while functions with the keyword "get" are convenience functions that target a whole list of trajectories at once
# for example, computePathLength() requires a single trajectory array, computes the path length for a single trajectory and returns a single number. 
# Contrary, getPathLength() requires a list of trajectories, computes the path length for every one of them and returns an array of the associated numbers

# load libraries
import numpy as np
import rdp # path simplification algorithm for flight scaled data
import math
import traja as trj # traja package (https://github.com/traja-team/traja)
import pandas as pd
from scipy import stats
from scipy import spatial as sp



#%%
# functions on step length

def computePathLength(path, dimension = "2d"):
    """Returns the total length of the path."""
    pathLength = np.sum(computeStepLength(path, dimension))
    return pathLength

def getPathLength(paths, dimension = "2d"):
        """Returns the total length of the path."""
        res = []
        for p in paths:
            res.append(computePathLength(p, dimension))
        return np.asarray(res)


# single path based functions with step wise return:
# relies on euclidian distance
def computeStepLength(path, dimension = "2d"):
    """Returns sequence of step lengths for a single path as a numpy array. """
    if dimension == "2d":
        if np.shape(path)[1] > 2: # consider possible path types (4d, 3d or 2d)
            path2d = path[:,[0,2]]
        elif np.shape(path)[1] == 2:
            path2d = path
        res = np.sqrt(np.sum((path2d[:-1]-path2d[1:])**2,axis=1))
    elif dimension == "3d":
        res = np.sqrt(np.sum((path[:-1,:]-path[1:,:])**2,axis=1))
    elif dimension == "y":
        res = np.sqrt(np.sum((path[:-1,[1]]-path[1:,[1]])**2,axis=1))
    return res

def getHistoryStepLength(paths, dimension = "2d", includeZeroSteps = True):
     """Returns the sequence of step lengths for each paths. Set includeZeroSteps to FALSE if only non-zero steps should be included."""
     res = []   
     for p in paths:
         stepHistory = computeStepLength(p, dimension)
         if includeZeroSteps == False:
             stepHistory = stepHistory[stepHistory > 0]
         res.append(stepHistory)
     return res


def computeLagStepPercent(stepHistory, cutoffMultiplier):
    
    stepHistoryNoZeroSteps = stepHistory[stepHistory > 0] # take median of non zero steps only 
    cutoffValue = np.median(stepHistoryNoZeroSteps) * cutoffMultiplier 
    lagSteps = stepHistoryNoZeroSteps[stepHistoryNoZeroSteps > cutoffValue]
    percentLagSteps = len(lagSteps) / len(stepHistory) # percentage of lag steps of all steps (zero steps included)
    return(percentLagSteps)

def getLagStepPercent(stepHistories, cutoffMultiplier):
    """Computes get percentage of lag steps given a predefiend cutoff. """  
    
    res = []
    for myHistory in stepHistories:
        percentLagSteps = computeLagStepPercent(myHistory, cutoffMultiplier)
        res.append(percentLagSteps)
    return(np.asarray(res)) 



#%%
# functions on time

def computeTime(path):
    """Returns the time spent exploring. Note that "time" is represented as the number of datapoints rather than an actual time unit, so the interpretation relies on the sampling rate. """
    return len(path)

def getTime(paths):
        """Returns the time spent exploring for a list of trajectories as a numpy array. Note that "time" is represented as the number of datapoints rather than an actual time unit, so the interpretation relies on the sampling rate."""
        res = []
        for p in paths:
            res.append(computeTime(p))
        return np.asarray(res)


# get raw number or percentage of events without movement in any of the possible dimensions
def computeTotalIdleTime(path, mode, dim): 
    steps = computeStepLength(path, dim)
    if mode == "raw":
        idle_time = len(path) - np.count_nonzero(steps) # raw number of steps without any movement
    elif mode == "percent":    
        idle_time = 1 - (np.count_nonzero(steps)/len(path)) # percentage of steps without any movement
    return idle_time

def getTotalIdleTime(paths, mode, dim):
    """Returns the number or the percentage of steps (indicated by mode = "raw" or mode = "percent") without movement. """
    # Note: trimming of the path alters the resulting idle time.
    res = []
    for p in paths:
        res.append(computeTotalIdleTime(p, mode, dim))
    return np.asarray(res)

def computeIdlePeriods(path): 
    """Returns a 2d array with the start and end points of all idle time periods"""
    tmp = computeStepLength(path, "3d")
    iszero = np.concatenate(([0], np.equal(tmp, 0).view(np.int8), [0])) # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    absdiff = np.abs(np.diff(iszero)) # give start of zero sequence as well as the first nonzero element after the sequence
    periodRanges = np.where(absdiff == 1)[0].reshape(-1, 2) # get an array with the index ranges of all 0 sequences 
    return periodRanges

def getIdleTimePeriods(paths):
    """Returns all idle time periods for all trajectories."""
    # Note: trimming of the path alters the resulting idle time.
    res = []
    for p in paths:
        res.append(computeIdlePeriods(p))
    return np.asarray(res)

def computeNumIdlePeriods(path, duration):
    """Returns the number of idle time periods, given a minimal duration (duration is measured in data points logged)"""
    periodRanges = computeIdlePeriods(path)
    periodDurations = periodRanges[:,1] - periodRanges[:,0]
    periodCount = len(np.where(periodDurations >= duration)[0]) # get number of idle times with durations >= the predefined Duratiom
    return periodCount

def getNumIdleTimePeriods(paths, duration):
    """Returns the count of idle time periods with a duration greater than minDuration."""
    # Note: trimming of the path alters the resulting idle time.
    res = []
    for p in paths:
        res.append(computeNumIdlePeriods(p, duration))
    return np.asarray(res)

def computeMaxIdlePeriod(path):
    """Returns the length of the longest idle time period"""
    periodRanges = computeIdlePeriods(path)
    periodDurations = periodRanges[:,1] - periodRanges[:,0]
    maxDuration = np.max(periodDurations)
    return maxDuration

def getMaxIdleTimePeriod(paths):
    """Returns the the length of the longest idle time period."""
    # Note: trimming of the path alters the resulting idle time.
    res = []
    for p in paths:
        res.append(computeMaxIdlePeriod(p))
    return np.asarray(res)


def deleteIdlePeriodsfromTraj(path, duration):
    """takes a single trajectory and deletes all idle time periods greater than a given duration (duration is measured in data points logged). Returns the trimmed trajectory as a numpy array."""
  
    # get idle time periods
    periodRanges = computeIdlePeriods(path)
    # get all idle sequences >= duration
    periodDurations = periodRanges[:,1] - periodRanges[:,0]
    periodFilter = np.where(periodDurations >= duration)[0]
    periodRangesFiltered = periodRanges[periodFilter,]
    
    # create filter for the original trajectory
    filterTrajectory = np.array([])
    for i in range(len(periodRangesFiltered)):
        mySequence = np.arange(periodRangesFiltered[i,0],periodRangesFiltered[i,1])
        filterTrajectory = np.append(filterTrajectory, mySequence)
    
    # delete all zero sequences
    pathTrimmed = np.delete(path, filterTrajectory.astype(int), axis = 0)
    return pathTrimmed
    
def deleteIdlePeriodsfromTrajGroup(paths, duration):
    """takes a group of trajectories and deletes all idle time periods greater than a given duration (duration is measured in data points logged).
        Returns the trimmed trajectories as a list of numpy arrays."""
    res = []
    for p in paths:
        res.append(deleteIdlePeriodsfromTraj(p, duration))
    return res 


#%%
# functions on areas


# gives area covered either as raw number of bins visited (if k = 1) or as percentage of bins visited (if k != None)
# k is defined as the total number of accessible bins (normalizes all values to be able to compare VEs of different sizes)
# if k is unknown, it may be approximated via:
# - the number of bins inside the respective bounding box
# - the number of bins accessed by at least one subject


def computeAreaCovered(paths, binSize, k = 1):   
    """Returns the total area covered by the path. Use k > 1, if you want to get a percentage of bins cavered rather than an absolute value (default: k = 1). """
    res = np.zeros((len(paths)))
    binNum = defineHist(binSize, paths)[0]
    bbox = defineHist(binSize, paths)[1]
    for i in range(len(paths)):
        res[i] = np.sum(computeHistogram([paths[i]], binNum, bbox)>0)/k # counts all bins where value is > 0 and norms by k
    return res


# roaming entropy
# k is defined as the total number of accessible bins (guarantees a normalization of entropy values to be able to compare VEs of different sizes)
# if k is unknown, it may be approximated via:
# - the number of bins inside the respective bounding box
# - the number of bins accessed by at least one subject
def computeRoamingEntropy(paths, binSize, k):
    """Returns the total area covered by the path."""
    res = np.zeros((len(paths)))
    binNum = defineHist(binSize, paths)[0]
    bbox = defineHist(binSize, paths)[1]
    for i in range(len(paths)):
        hist = computeHistogram([paths[i]], binNum, bbox) # compute histogram with frequencies
        histProbs= hist / np.sum(hist) # convert frequencies to probabilities
        # calculate roaming entropy
        # for undefined values: if probability is zero, set log2(prob) to zero (Ref: https://stats.stackexchange.com/questions/57069/alternative-to-shannons-entropy-when-probability-equal-to-zero)
        res[i] = -np.sum(histProbs*(np.where(histProbs>0, np.log2(histProbs), 0))) / math.log2(k) 
    return res


def computeHistogram(paths, bins, bbox):
    """Returns a histogram of the trajectory, representing all datapoints logged in the given bins."""
    hist = np.histogram2d(paths[0][:,0], paths[0][:,2], bins, bbox)[0]
    for p in paths[1:]:
        hist += np.histogram2d(p[:,0], p[:,2], bins, bbox)[0]
    return hist


def computeHistBinsEntered(paths, bins, bbox):
    """Returns a histogram of the trajectory, where bins entered/undiscovered are coded with 1/0 respectively """
    res = []
    for i in range(len(paths)):
        myhist = computeHistogram([paths[i]], bins, bbox)
        mymask = (myhist > 0).astype(int) # set visited bins to one 
        res.append(mymask)
    return res




#%%
# functions on minimum complex polygon / activity space

def computeMinimumPolygon(path):
   """Returns the area of the smallest convex polygon /smallest convex hull covering all datapoints of a single trajectory.
   Requires a 2d trajectory."""   
   hull = sp.ConvexHull(path)
   hullArea = hull.volume
   return hullArea

def getMinimumPolygon(paths):
   """Returns the area of the smallest convex polygon /smallest convex hull covering all datapoints of a group of trajectories.
   Requires 2d trajectories."""   
   res = []
   for p in paths:
       res.append(computeMinimumPolygon(p))
   return np.asarray(res)





#%%
# functions on revisiting

def computeRevisiting(path, radius, method = "Gagnon", refractoryPeriod = 10):
    """Returns the Revisiting score for a single trajectory.  2d path required.
    
    'radius' specifies the radius of the circle in which revisits are looked for
    
    method = 'Gagnon' computes Revisiting exactly as given in Gagnon et al., 2016, where all points in radius following the first revisit are classified as Revisits
    
    method = 'alternative' classifies only the first point in the radius after a out-of radius sequence as a revisit 
    
    The 'refractoryPeriod' allows to specify how man steps the participant needs to move of the radius until moving inside the radius counts as another revisit (defaults to 1s, as in Gagnon et al., 2016) """

    if np.shape(path)[1] > 2:
        raise Exception('Please provide 2d path')
      
    numRevisitsPointwise = []
    
    for currentPos in np.arange(len(path)): # loop across all points in the trajectory
        
        currentCoord = np.reshape(path[currentPos],[1,2]) # get current position
        previousCoords = np.flip(path[:currentPos], axis = 0) # get all point prior to the current position, orderd from most previous to least previous point
        distances = np.sqrt(np.sum((previousCoords - currentCoord)**2,axis=1)) # calcuate all distances between current position and all previous positions
        indicesDistancesInRadius = np.where(distances <= radius)[0] # get indices of all distances within the radius
        indexDiffs = np.diff(indicesDistancesInRadius) # get differences between indices
        indicesDiffsRevists = np.where(indexDiffs > refractoryPeriod)[0] # get indices of all "Revists" (at least X prior distances out of radius) out of all distances in radius
    
    
        if  indicesDiffsRevists.size  != 0:
            if method == "Gagnon":
                # Gagnon method
                indexFirstRevist = np.min(indicesDiffsRevists) # get index of the first Revist
                indicesRevisitDistances = indicesDistancesInRadius[indexFirstRevist + 1:len(indicesDistancesInRadius)] # get all distances in radius after the first revisit (add 1 to start to match indexing)
                revisitDistances= distances[indicesRevisitDistances] # get the distances classified as "revisits"
                #revisitPoints = previousCoords[indicesRevisitDistances + 1]# get the actual  points classified as "revisits"
                numRevisits = len(revisitDistances)
        
            elif method == "alternative":
                # alternative proposal (this classifies only the initial "revisit" points as revisits, not all points in radius after the first revisit)
                indicesRevisitDistancesAlt = indicesDistancesInRadius[indicesDiffsRevists + 1]
                revisitDistancesAlt =  distances[indicesRevisitDistancesAlt] # get the distances classified as "revisits"
                #revisitPointsAlt = previousCoords[indicesRevisitDistancesAlt + 1]# get the actual  points classified as "revisits"
                numRevisits = len(revisitDistancesAlt)
        
        else:
            numRevisits = 0
        
        numRevisitsPointwise.append(numRevisits)
    
    meanRevisiting = np.mean(np.asarray(numRevisitsPointwise))
    
    return(meanRevisiting)



def getRevisiting(paths, radius, method = "Gagnon", refractoryPeriod = 1):
    """Returns the Revisiting score for a list of trajectories.  2d paths required.
    
    method = 'Gagnon' computes Revisiting exactly as given in Gagnon et al., 2016, where all points in radius following the first revisit are classified as Revisits
    
    method = 'alternative' classifies only the first point in the radius after a out-of radius sequence as a revisit 
    
    The 'refractoryPeriod' allows to specify how man steps the participant needs to move of the radius until moving inside the radius counts as another revisit (defaults to 1, as in Gagnon et al., 2016) """

    res = []

    for p in paths:
        meanRevisiting = computeRevisiting(p, radius = radius, method = method, refractoryPeriod = refractoryPeriod)
        #print(meanRevisiting)
        res.append(meanRevisiting)
    
    return(np.asarray(res))



#%%
# functions on objects



# uses euclidian distance
def computeObjDistStepwise(path, obj):
    """computes an array with the euclidian distance of a given trajectory to one given object across time.
    CAVE: requires a 2d trajectory and 2d object coordinates.""" 
    return np.sqrt(np.sum((path-obj)**2,axis=1))

def computeObjVisits(path, objects, maxDistance, minDuration, minTimeRevisit):
    """returns a list of arrays, with each array representing visits for one object.
    Use maxDistance to specify the radius around the objects.
    Use minDuration > 0, if a vsisit additionally should be defined by the time spent within the object radius.
    Use minTmeRevisit to set a period, where Revisits are not counted (after an individual left the object radius).
    CAVE: requires a 2d trajectory and 2d object coordinates."""         
    
    list_periods= []
    objDistances = np.zeros((len(path),1))
    for i in range(len(objects)):
            
            objDistances = computeObjDistStepwise(path, objects[i,:])

            mask_array = objDistances <= maxDistance # array is TRUE whenever position is within maxDistance around object
            mask_array = mask_array.astype(int) # convert bools to int
            iszero = np.concatenate(([0], mask_array, [0])) # pad a zero at each end
    
            absdiff = np.abs(np.diff(iszero)) # give start of zero sequence as well as the first nonzero element after the sequence
            periodRanges = np.where(absdiff == 1)[0].reshape(-1, 2) # get an array with the index ranges of all 0 sequences 
 
            periodDuration = periodRanges[:,1] - periodRanges[:,0] # calculate duration of each period
            periodRanges = periodRanges[periodDuration >= minDuration] # select only periods with a duration > minDuration

            periodDistance = periodRanges[1:,0] - periodRanges[0:-1,1] # calculate distances between periods
            periodRanges = periodRanges[1:][periodDistance >= minTimeRevisit] # select only periods with a distance to the previous period > minTimeRevisit
            
            list_periods.append(periodRanges)
            
    return list_periods



def getNumObjVisited(paths, objects, maxDistance, minDuration, minTimeRevisit):
    """returns the number of objects visited at least once. 
    Use maxDistance to sepcify the radius around the objects.
    Use minDuration > 0, if a vsisit additionally should be defined by the time spent within the object radius.
    Use minTmeRevisit to set a period, where Revisits are not counted (after an individual left the object radius)."""        
    totalVisits = [] 
    for p in paths:
        objPeriods = computeObjVisits(p, objects, maxDistance, minDuration, minTimeRevisit)
    
        objVisits = []
        for obj in objPeriods:
            countPeriods = len(obj) 
            # if object was visited write 1, of not write 0
            if countPeriods > 0: 
                objVisits.append(1) 
            if countPeriods == 0: 
                objVisits.append(0)   
                 
        sumVisits = sum(objVisits)
        totalVisits.append(sumVisits) # stores all object periods 
    
    totalVisits = np.asarray(totalVisits) 
    return totalVisits


def getNumObjVisits(paths, objects, maxDistance, minDuration, minTimeRevisit):
    """returns the total number of object visits, including revisits.   
    Use maxDistance to sepcify the radius around the objects.
    Use minDuration > 0, if a vsisit additionally should be defined by the time spent within the object radius.
    Use minTmeRevisit to set a period, where Revisits are not counted (after an individual left the object radius)."""             
    totalPeriods = [] 
    for p in paths:
        objPeriods = computeObjVisits(p, objects, maxDistance, minDuration, minTimeRevisit)
    
        countPeriodsList = []
        for obj in objPeriods:
            countPeriods = len(obj) # period of one object 
            countPeriodsList.append(countPeriods) # stores all object periods for one subject
            
        sumPeriods = sum(countPeriodsList)
        totalPeriods.append(sumPeriods) # stores all object periods 
    
    totalPeriods = np.asarray(totalPeriods) 
    return totalPeriods




#%%
# functions on angles
# this is based on 2d direction vectors (from x and z coordinates) 
# inspired by: https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2

# gives changes in turning angles in degrees
def computeAngles(path2d, span=1, angleType = "degree"):
    """Compute an array representing the sequence of turning angles. 
    Use span to define if angles are computed from step to step (span = 1), or if steps should be omiited (giving higher vs. lower frequency turning angles).
    Use angleType to specify if the output is in degrees or radians (default is degrees)..
    Trajectoires need to be provided as 2d arrays (x and z coordinates). 
    Note that angles are calculated as absolute values (neglects left/right turning)."""
    
    if np.shape(path2d)[1] > 2:
        raise Exception('Please check if 2d path is provided.')
    
    mypath = path2d[::span].copy() # path subsample based on "span" 

    vectors = mypath[1:,]-mypath[:-1,]

    dotProd = vectors[1:,0] * vectors[:-1,0] + vectors[1:,1] * vectors[:-1,1] # get dot product for two consecutive vectors
    magA = (vectors[1:,0] * vectors[1:,0] + vectors[1:,1] * vectors[1:,1]) **0.5
    magB = (vectors[:-1,0] * vectors[:-1,0] + vectors[:-1,1] * vectors[:-1,1]) **0.5
    magnitudeProduct = magA * magB
    cos = np.divide(dotProd, magnitudeProduct) # cosine formula: cos = dotproduct(V1 * V2) / mag(V1) * mag(V2). Note: this will give division by zero erros, if steps without changes in angle occur
    cos = np.around(cos, 3) # we will round the cosine values, since otherwsie some values are invisibly prone to rounding errors in np.divide() (i.e, values of 1 are actually stored as 0.999999...)
    angles = np.arccos(cos) # get angle via arccosine (note that this also we give errors, if steps without changes in angle occur)
    
    if angleType == "degree":
        angles = np.degrees(angles) # convert radians to degrees 
    
    angles = np.insert(angles, 0, 0) # pad a zero to the beginning to account for the first step (so step history and angle history have the same lenght)
    
    #angles_deg = np.where(np.isnan(angles_deg), 0, angles_deg) # replace nan values if necessary
    return angles


#  get list of angle array for multiple paths
def getHistoryAngles(paths, span=1, angleType = "degree"):
    ''' Return a list of arrays, where each array represents the sequence of turning angles. 
    Use span to define if angles are computed from step to step (span = 1), or if steps should be omiited (giving higher vs. lower frequency turning angles).
    Use angleType to specify if the output is in degrees or radians (default is degrees).
    Trajectoires need to be provided as 2d arrays (x and z coordinates). 
    Note that angles are calculated as absolute values (neglects left/right turning).''' 
    res = []
    for p in paths:
        res.append(computeAngles(p, span, angleType))
    return res

# use angle computation from the traja package for crossvalidation
def getHistoryAnglesTraja(traja_paths, absolutValues = "yes"):
    res = []
    for p in traja_paths:
        mytraj = trj.trajectory.calc_turn_angle(p) # get turning angle
        mytraj = np.asarray(mytraj) # convert to numpy array
        
        if absolutValues == "yes": # get absolute values if desired (neglects left/right turning) 
            mytraj = np.absolute(mytraj) 
        
        res.append(mytraj)
    return res


# number of "valuable turning points"
# similar to here: https://www.gakhov.com/articles/find-turning-points-for-a-trajectory-in-python.html
def computeTurningPoints (path, span, minAngle, angleType):
    """Counts all angles above a predefined value (minAngle). 
    CAVE: check that minAngle is the same unit as specified in angleType. 
    Use span to define if angles are computed from step to step (span = 1), or if steps should be omiited (giving higher vs. lower frequency turning angles).
    Use angleType to specify if the output is in degrees or radians (default is degrees).
    Trajectoires need to be provided as 2d arrays (x and z coordinates). 
    Note that angles are calculated as absolute values (neglects left/right turning). """
    myAngles = computeAngles(path, span, angleType=angleType)
    turningPoints = myAngles[myAngles >= minAngle]
    return turningPoints

def getNumTurningPoints (paths, span, minAngle, angleType):
    """returns a list of counts of all angles above a predefined value (minAngle). 
    CAVE: check that minAngle is the same unit as specified in angleType. 
    Use span to define if angles are computed from step to step (span = 1), or if steps should be omiited (giving higher vs. lower frequency turning angles).
    Use angleType to specify if the output is in degrees or radians (default is degrees).
    Trajectoires need to be provided as 2d arrays (x and z coordinates). 
    Note that angles are calculated as absolute values (neglects left/right turning). """
    res = []
    for p in paths:
        turningPoints = computeTurningPoints(p, span, minAngle, angleType)
        numTurningPoints = len(turningPoints)
        res.append(numTurningPoints)
    return res


#%%
# Sinuosity

def computeSinuosity(path, rediscretized):
    """ computes sinuosity for a given path. 
    if the traejctory has steps of equal lengths or has been redidcretized, set rediscretized = True, which then uses the simple formula (Bovet & Benhamou, 1988).
    if steps are of unequl length, set rediscretized = False, which then applies the corrected formula (Benhamou, 2004)."""   
    
    if np.shape(path)[1] > 2:
        raise Exception('2d path needed. Please check if 2d path is provided.')
        
    stepLengths = computeStepLength(path) 
    meanStepLength = np.nanmean(stepLengths)
    stDevStepLength = np.nanstd(stepLengths, ddof=1) # use Bessel's correction for sample variance, to make results comparabel to R package trajr
    CoeffVar =  stDevStepLength / meanStepLength # Coefficient of Variation for step length
    
    angles = np.deg2rad(getHistoryAnglesTraja(NumpyToTraja([path], num_cols = 2), absolutValues="no")[0]) # get angles in radians
    stDevAngles = np.nanstd(angles, ddof=1) # use Bessel's correction for sample variance, to make results comparabel to R package trajr
    meanCosine = np.nanmean(np.cos(angles)) # mean cosine of turning angles

    if rediscretized == True:
        sinuosity = 1.18 * (stDevAngles / (math.sqrt(meanStepLength))) # sinuosity formula for regular step length
        if stDevAngles > 1.2:
            raise Exception('Be Careful: Sinuosity may not work correctly with sd(angle) > 1.2. See Bovet & Benhamou (1988) for details')
    else:
        sinuosity = 2 / math.sqrt(meanStepLength * (((1+meanCosine) / (1-meanCosine)) + CoeffVar ** 2)) # sinuosity formula for irregular step lengths

        
    return sinuosity



def getSinuosity(paths, rediscretized):
    """ returns sinuosity values for multiple paths as numpy array. 
    Specifiy "rediscretize" to select sinuposity formula dependent on trajectory type -> see computeSinuosity().
    CAVE: may take a long time to compute!"""   

    res = []
    
    for p in paths:
        sinuosity = computeSinuosity(p, rediscretized = rediscretized)
        res.append(sinuosity)

    return np.asarray(res)






#%%
# fractal dimension

def CreateLogSequence(start, stop, numSteps):
    """Use this to get log-distributed steps for the calcualtion of Fractal Dimension (see Nams, 2006).
    """
    stepSizes = np.linspace(np.log10(start), np.log10(stop), numSteps) # get a series of steps evenly distributed on the log scale
    stepSizes = np.exp(stepSizes * np.log(10))
    return stepSizes


def computeFractalDimensionValues(path, stepSizes, adjustD):
    """Returns a numpy array containing the Fractal Dimension Values for different stepSizes given in stepSizes. 
    StepSizes can be specified via CreateLogSequence().
    The path needs to be in traja fromat (x and y coordinates only, reprsented as pandas dataframe - you may use numpyToTraja() for the conversion.
    Set adjustD = True (default) to correct for truncation error according to Nams (2006)
    """
    fractalDimensionValues = np.empty((0,2), float)

    for myStepSize in stepSizes:

        pathRD = trj.rediscretize_points(path, myStepSize) # rediscretize to selected step size
        pathRDarray = TrajaToNumpy([pathRD]) # convert back to numpy array (returns a list)

        if adjustD == True: # adjust as suggested by Nams(2006)
        # compute distance from last datapoint of the original trajectory to the last datapoint of the rediscretized trajectory
            lastRD = pathRDarray[0][-1] 
            lastTrj = TrajaToNumpy([path])[0][-1]
            distCorrection = np.linalg.norm(lastRD-lastTrj) # get euclidian distance between points
            pathLength = computePathLength(pathRDarray[0], "2d") # get path length of rediscretized path, convert list back to array
            pathLength = pathLength + distCorrection # add correction 
        else: 
            pathLength = computePathLength(pathRDarray[0], "2d") # stm.computeLengtnh only accepts list input, so conversion back and forth is necessary

        fractalDimensionValues = np.vstack((fractalDimensionValues,  np.array([[myStepSize, pathLength]]))) # append results to fractalDimensionValues Array
    
    return fractalDimensionValues
    
    

def getFractalDimensionValues(paths, stepSizes, adjustD):
    """Returns a list of arrays with the Fractal Dimension Values for each path. 
    "Paths" needs to be a list of trajectories in traja fromat (x and y coordinates only, as pandas dataframe). You may use numpyToTraja() for the conversion.
    StepSizes can be specified via CreateLogSequence().
    Set adjustD = True (default) to correct for truncation error according to Nams (2006)."""
    res = []
    
    for p in paths:
        fractalDimensionValues = computeFractalDimensionValues(p, stepSizes = stepSizes, adjustD = adjustD)
        res.append(fractalDimensionValues)
        
    return res



def computeFractalDimension(path, stepSizes, adjustD = True, meanD = True):
    """ 
    Compute the Fractal Dimension for a single trajectory.
    Input needs to be a dataframe compatible to traja (x and y coordinates only, as pandas dataframe). You may use numpyToTraja() for the conversion.
    If meanD = True, dividers are walked forwards as well as backward to correct for rediscretization biases (Nams, 2006). Defaults to True.
    adjustD = True corrects for truncation error (Nams, 2006). Defaults to True.
    """  

    FDvalues = computeFractalDimensionValues(path, stepSizes, adjustD = adjustD)
    mymodel = stats.linregress(x = np.log(FDvalues[:,0]), y = np.log(FDvalues[:,1])) 
    slope = mymodel.slope
    fractalDim = 1 - slope   
        
    if meanD == True:
        pathRev = path[::-1] # reverse path
        FDvaluesRev = computeFractalDimensionValues(pathRev, stepSizes, adjustD = adjustD) 
        mymodel = stats.linregress(x = np.log(FDvaluesRev[:,0]), y = np.log(FDvaluesRev[:,1]))
        slopeRev = mymodel.slope
        fractalDimRev = 1 - slopeRev   
        
        fractalDim = np.mean(np.asarray([fractalDim, fractalDimRev])) # correcte fractal dimension is mean of forwards and backwards FD
        
    return fractalDim


def getFractalDimension(paths, stepSizes, adjustD, meanD):
    """ 
    Compute the Fractal Dimension for a list of trajectories, returns values as a single numpy array.
    Input needs to be a dataframe compatible to traja (x and y coordinates only, as pandas dataframe). You may use numpyToTraja() for the conversion.
    If meanD = True, dividers are walked forwards as well as backward to correct for rediscretization biases (Nams, 2006). Defaults to True.
    adjustD = True corrects for truncation error (Nams, 2006). Defaults to True.
    CAVE: may take along time to compute!
    """  
    res = []
    
    for p in paths:
        
        FD = computeFractalDimension(p, stepSizes, adjustD, meanD)
        res.append(FD)
        
    return(np.asarray(res))



#%%
# path simplification


def simplifyPath(path, epsilon):
    """Applies the rdp algorithm to a single path (needs to be a numpy array) to rescale the trajectory to the flight scale.
    Returns the simplified path as numpy array.
    Use Epsilon to specify the degree of simplification (higher epsilon leads to greater simplification). 
    See the rdp package for more information."""
    return rdp.rdp(path, epsilon)

def getSimplePaths(paths, epsilon=5):
    """Applies the rdp algorithm to a list of paths (needs to be a list of numpy arrays) to rescale the trajectories to the flight scale.
    Returns the simplified paths as a list of numpy arrays.
    Use Epsilon to specify the degree of simplification (higher epsilon leads to greater simplification). 
    See the rdp package for more information.
    CAVE: may take a long time to compute!"""
    
    res = []
    for p in paths:
        simplePath = simplifyPath(p, epsilon)
        res.append(simplePath)
    return res

def computeCompactness(paths_flightScaled):
     """Returns the compression rate/the number of remaining vertices of a list of flight scaled paths (as numpy array). Needs a list of flight scaled paths as input.
     CAVEAT: Result depends heavily on epsilon and does NOT scale linearly with varing epsilon."""
     res = []
     for p in paths_flightScaled:
         res.append(len(p))
     res = np.asarray(res)
     return res




#%%
# helper functions


####### functions for building histograms of adaptive size


def computeBoundingBox(paths):
    """Finds the edges of the histogram, given a group of trajectories in the same environment (minimal and maximal x and z values of all paths). 
    Input needs to be a list of numpy arrays.
    returns a list containing two arrays (minimal and maximal x values, minimal and maximal z values).
    CAVE: requries a 3d trajectory"""
    xmin,ymin,zmin = np.min(paths[0], axis=0) # get min for each axis
    xmax,ymax,zmax = np.max(paths[0], axis=0) # get max for each axis
    for p in paths:
        xmin,ymin,zmin = np.min([[xmin,ymin,zmin], np.min(p, axis=0)],axis=0)
        xmax,ymax,zmax = np.max([[xmax,ymax,zmax], np.max(p, axis=0)],axis=0)
    return [[math.floor(xmin),math.ceil(xmax)],[math.floor(zmin),math.floor(zmax)]]


def computeBinNumber(binSize, bbox):
    """Given a predefined bin size (binSize) and the environment size (bbox), get the optimal number of bins to build the histogram.    
    """
    # get min and max values for each direction
    xmin = bbox[0][0]
    xmax = bbox[0][1]
    zmin = bbox[1][0]
    zmax = bbox[1][1]

    # get max x and z distances (rounded to the next higher int)
    xDist = abs(xmin - xmax)
    zDist = abs(zmin - zmax)
    
    # compute number of bins for each axis. 
    xBinNum = xDist / binSize
    zBinNum = zDist / binSize
    
    # If bins don't fit as a whole, increase size of the bounding box until bins fit nicely.
    while xBinNum.is_integer() == False:
        xDist = xDist + 1
        xBinNum = xDist / binSize

    while zBinNum.is_integer() == False:
        zDist = zDist + 1
        zBinNum = zDist / binSize      
        
    return[int(xBinNum), int(zBinNum), xDist, zDist] # return binNum as int, since np.histogram2d() needs 'bins' argument in integeres  



def refitBoundingBox(bbox, binSize):
    """Rescales the bounding box created by computeBoundingBox so the bins fit nicely inside.
    Specify the bounding box (bbox) by using computeBoundingBox().
    Specify the size of the bins (binSize).
    """
    xmin = bbox[0][0]
    xmax = bbox[0][1]
    zmin = bbox[1][0]
    zmax = bbox[1][1]
    
    xDistOriginal = abs(xmin - xmax)
    zDistOriginal = abs(zmin - zmax)
     
    xDistNew =  computeBinNumber(binSize, bbox)[2]
    zDistNew =  computeBinNumber(binSize, bbox)[3]
    
    xDiffOriginalNew = xDistNew - xDistOriginal 
    zDiffOriginalNew = zDistNew - zDistOriginal 
    
    xminNew = xmin - xDiffOriginalNew/2 
    xmaxNew = xmax + xDiffOriginalNew/2
    zminNew = zmin - zDiffOriginalNew/2 
    zmaxNew = zmax + zDiffOriginalNew/2
    
    bboxNew = [[xminNew, xmaxNew], [zminNew,zmaxNew]]
    
    return bboxNew




def defineHist(binSize, paths, world = None):
    ''' takes in the desired bin size and returns a list containing the number of bins the map will be divided into as well as the map endpoints that span the grid (upper left and lower right corner points).'''
    # get initial world limits
    path_data_resampled_3d = convert4dTo3d(paths)
    
    # if world == "green":
    #     worldLimits = stm.computeBoundingBox([path_data_resampled_3d[i] for i in f_green])
    # elif world == "pink": 
    #     worldLimits = stm.computeBoundingBox([path_data_resampled_3d[i] for i in f_pink])

    worldLimits = computeBoundingBox(path_data_resampled_3d)
    
    # get the number of bins needed to cover area with the given worldLimits 
    bins = computeBinNumber(binSize, worldLimits)[0:2] # gives the number of bins for each axis in the 2d histogram
    worldLimitsNew = refitBoundingBox(worldLimits, binSize) # if necessary, extend world limits so bins fit nicely inside
    
    return [bins, worldLimitsNew]




##### convert trajectories from one dimension to another

def removeHeigthCoordinate(path):
    return path[:,[0,2]]

# convert 3d path to 2d path
def convert3dTo2d(paths):
    """ Convert a group of 3d trajectories to 2d trajectories. Requires a list of numpy array with three columns, returns a list of numpy array with three columns.
    CAVE: assumes that trajectories are in a x,y,z order, where y is upwards/downwards movement. """
    res = []
    for p in paths:
        res.append(removeHeigthCoordinate(p))
    return res

# convert 3d path to 2d path
def convert4dTo3d(paths):
    """ Convert a group of 3d trajectories to 2d trajectories. Requires a list of numpy array with four columns, returns a list of numpy array with three columns.
    CAVE: assumes that trajectories are in a x,y,z,time order, where y is upwards/downwards movement. """
    res = []
    for p in paths:
        p3d = p[:,[0,1,2]]
        res.append(p3d)
    return res

def convert2dTo3d(paths, height = 1):
    """ Convert a group of 2d trajectories to 3d trajectories. Requires a list of numpy array with two columns, returns a list of numpy array with three columns.
    CAVE: assumes that trajectories are in a x,z order. "Height"" specifies the middle "y" column that will be inserted. By default this can be a fixed value, or a numpy array with a single column."""   
    res = []
    for p in paths:      
        yCoordinate = np.zeros(len(p)).reshape(len(p),1) # create a dummy y coordinate
        yCoordinate[:] = height 
        path3d = np.hstack((p, yCoordinate)) # add ycoordinate to original array
        path3d[:, [2, 1]] = path3d[:, [1, 2]] # rearrange array so everything is in x,y,z order again
        
        res.append(path3d) # append to return list
        
    return res


def convert3dTo4d(paths):
    
    res = []
    for p in paths:
        
        yCoordinate = np.zeros(len(p)).reshape(len(p),1) # create a dummy y coordinate
        path4d = np.hstack((p, yCoordinate)) # add y coordinate to original array
        
        #path3d[:, [3, 1]] = path3d[:, [1, 3]] # swap columns to correct order (x,y,z,time)
        #path3d[:, [3, 2]] = path3d[:, [2, 3]]
        path4d[:, [1, 2, 3]] = path4d[:, [3, 1, 2]]
        
        res.append(path4d) # append to return list
        
    return res



# if desired, use this to smooth out the histogram data
def smoothTriangle(data, degree):

    # From: https://plotly.com/python/smoothing/
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed



def DescribeHistory(Histories, SummaryStat):
    """returns array with summary stats on chosen history data (angles, speed, minObjDistance..). Requires a list of numpy arrays (i.e., a list with arrays each containing the step lengths across time for a single trajectory)"""  
    res = []
    for p in Histories:
        if SummaryStat == "mean":
            res.append(np.nanmean(p))
        elif SummaryStat == "median":
            res.append(np.nanmedian(p))
        elif SummaryStat == "stdev":
            res.append(np.nanstd(p))
        elif SummaryStat == "max":
            res.append(np.max(p))
        elif SummaryStat == "min":
            res.append(np.min(p))
        elif SummaryStat == "CoeffVar":
            res.append(np.nanstd(p) / np.nanmean(p))               
    return np.asarray(res)


def describeDistribution(df):
    """returns array with descriptive stats on chosen dataframe column (angles, speed, minObjDistance..). Requires a list of numpy arrays (i.e., a list with arrays each containing the step lengths across time for a single trajectory)"""  
    res = []
    coeffVar = df.std() / df.mean()
    skewness = df.skew()
    kurtosis = df.kurtosis()
    # get percent of cases in ceiling/floor range
    valueRanges = df.max() - df.min() # get ranges of all parameter values 
    rangeSteps = valueRanges / 100 # get percent steps 
    floorRange =  df.min() + 10*rangeSteps 
    ceilingRange = df.max() - 10*rangeSteps
    floorCases = df[df <= floorRange].count() # get count of cases in floor Range
    floorCasesPercent = floorCases / df.shape[0] * 100 # get percentage of cases in floor Range
    ceilingCases = df[df >= ceilingRange].count() # get count of cases in ceiling Range
    ceilingCasesPercent = ceilingCases / df.shape[0] * 100 # get percentage of cases in ceiling Range

    res = pd.DataFrame({"coefficient of variation": coeffVar, "skewness": skewness, "kurtosis": kurtosis, "percent cases in lowest 10% of values": floorCasesPercent, "percent cases in highest 10% of values": ceilingCasesPercent})
    res = res.round(3)
    return res


def standardizeHist(hist, method, N = 1):
    if method == "divN":
        hist_standardized = hist / N
    elif method == "Zscore":
        hist_standardized = (hist - np.mean(hist)) / np.std(hist)
    return hist_standardized



def NumpyToTraja(paths, num_cols):
    res = []
    
    for p in paths:
        
        if num_cols == 3:
            mypath = p[:,[0,2]] # extract x and z
        elif num_cols == 2:
            mypath = p
            
        mydf = pd.DataFrame(mypath, columns = ['x','y'])
        mytraj = trj.TrajaDataFrame(mydf)
        
        res.append(mytraj)
    
    return res


def TrajaToNumpy(paths):
    res = []
    for p in paths:
        res.append(np.asarray(p))
    return res


def rediscretizePaths(paths, R):
    """redicretize path to a fixed step lenght. requires path to be in traja input format"""
    res = []
    for p in paths:
        res.append(trj.rediscretize_points(p, R))
    return res



def create_timestamp(path, sessionDuration = "variable", samplingInterval = "unknown"):
    """Takes a path without timestamps (as array), returns the original path plus timestamps (as array).
    If the total duration is fixed across all trials and particiapnts, but the sampling rate is unknown, specify the session time in "session_duration" and set samplingInterval to "unknown".
    If the sampling rate is known specify samplingInterval as well as session_duration. 
    If the session duration varies across session/participants, set session_duration to "variable". """

    # decide whether to use fixed or variable session length
    if samplingInterval == "unknown" and sessionDuration != "variable":
        # estimate the samping interval based on the given duration
        samplingInterval =  sessionDuration / len(path)
        timeSequence = np.linspace(0, len(path)*samplingInterval, len(path)).reshape((len(path),1)) # reshape to specify dimensions
    
    elif samplingInterval != "unknown":
        if sessionDuration == "variable": # compute the duration given the specified sampling rate
            sessionDuration = len(path)
        sessionEndTime = round(sessionDuration * samplingInterval, 4) # compute endpoint (round to a specified number of decimals to prevent fuzzy float endpoints -> we use four here)
        timeSequence = np.arange(0,sessionEndTime, samplingInterval) # create the time sequence
        timeSequence = timeSequence.reshape((len(path),1)) # reshape into a format that can be appended to the trajectory array
        
    new_path = path
    new_path = np.append(path, timeSequence, axis = 1) # 
    return(new_path)

                 
def add_timestamps(paths, sessionDuration, samplingInterval):
    """add timestamps to all input paths (as a list of arrays), given either sampling intervals or session duration. """                
    res= []
    for p in paths:
        new_path = create_timestamp(p, sessionDuration, samplingInterval)
        res.append(new_path)
    return res


def resampleTime(path, stepTime):
    """ resample a single trajectory to a regular sampling interval given by 'steptime'. Needs a 2d path (x an z) plus a third timestamp column."""   
    new_times = np.array(np.arange(np.min(path[:,2]), np.max(path[:,2]) + stepTime, stepTime))
    x = np.interp(x = new_times, xp = path[:,2], fp = path[:,0]) # interpolate points
    z = np.interp(x = new_times, xp = path[:,2], fp = path[:,1]) # interpolate points
    path_resampled = np.vstack([x,z,new_times]).T# concat points back to a single array      
    return path_resampled



def resamplePaths(paths, stepTime):
    """ resample multiple trajectories to a common regular sampling interval given by 'steptime'. Needs 2d paths (x an z) plus a third timestamp column."""   
    res= []
    for p in paths:
        resampledPath = resampleTime(p, stepTime = stepTime)
        res.append(resampledPath)
    return(res)
        


def parameterIteration(parameterValues, evalFunction, argumentDict, parameterName):
    """ iterate across a function using different parameter values and save the output to a dataframe.
    parameterValues specifies the different values you want the function to iterate across (expects a list of values).
    evalFunction contains the function name of the function that you want to evaluate.
    argumentDict is a dictionary that contains all the function arguments necessary for the evalFunction.
    parameterName is a string representing the keyword for the parameter you want to iterate across.
    """   
     
    res = []
    for i in parameterValues:
        argumentDict[parameterName] = i
        currentData = evalFunction(**argumentDict)
        res.append(currentData)
        
    df = pd.DataFrame(res).T
    #df.columns = parameterValues
    return df


