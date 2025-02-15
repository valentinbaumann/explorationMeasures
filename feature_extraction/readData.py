
#### all functions related to loading the raw data

### import libraries and set paths

import numpy as np
import pandas as pd
import os



### define functions

def trimPath(path, dimension = "3d"):
    """Trims path from start to remove initial idle time from logs."""
    
    if dimension == "3d":
        step_diff = path[1:,[0,2]]-path[:-1,[0,2]] # get difference between steps (x and z axis only)
    elif dimension == "2d":
        step_diff = path[1:,[0,1]]-path[:-1,[0,1]] # get difference between steps (x and z axis only)
        
    idx = np.nonzero(step_diff)[0] # get index for all nonzero elements (x and z)
    res = np.copy(path[idx[0]:,:]) # create trimmed path from first nonzero element onwards

    return res


def trimPathSet(paths, dimension = "3d"):
    """Trims a set of paths from start to remove initial idle time from logs. 3d input only!"""

    res = []

    for p in paths:
            trimmed_path = trimPath(p, dimension)
            res.append(trimmed_path)
    return res

 
def cutPath(path, cutoff):
    shortPath = path[0:cutoff,:]
    return shortPath
    
def cutPathSet(paths, cutoff):
    res = []
    for p in paths:
        shortPath = cutPath(p, cutoff)
        res.append(shortPath)
    return res


def loadPersonData(pathToMainTable, filename):
    """Loads the dataset with all the information associated with the idnividual trajectories. Requires an .sav (SPSS) file as input."""    
    df_persons = pd.read_csv(os.path.join(pathToMainTable, filename))
    return df_persons




def loadLogsFromCSV(path, sep, header): 
    
    fileList = os.listdir(path) # get file list
    os.chdir(path)
    res = []
    
    print("reading files from: " + path)

    for p in fileList:
        mytraj_df = pd.read_csv(p, sep=sep, header=header)

        mytraj = mytraj_df.values

        res.append(mytraj)
        print(p)
    return res



def dropColumn(paths, columns):
    """ 'columns' defines the columns to delete (specify as list)"""    

    res = []
    for p in paths:
        reduced_p = np.delete(p, columns, axis = 1)
        res.append(reduced_p)
    return res

    
def addCoordOffset(paths, x_offset, z_offset):
    ''' requires 3d path'''
    res = []
    for p in paths:
        p[:,0] = p[:,0] + x_offset
        #p[:,2] = p[:,2] * -1 + z_offset
        p[:,2] = p[:,2] + z_offset
        res.append(p)
    return(res)
    

def saveFlightLogsToCSV(paths, df_persons, filePath):
    """add timestamps to all input paths (as a list of arrays), given either sampling intervals or session duration. """    
    # save to csv    
    #df_persons['Subject'] = df_persons['Subject'].astype(int).astype(str) # convert to in and then str to remove decimals
    numSubjects = len(df_persons)

    for i in range(numSubjects):
        trajectory = paths[i]
        logName = df_persons["Subject"].values[i]
        print(logName)

        filePathIndividual = filePath + "\\" + logName + "_flight" + ".csv"
        pd.DataFrame(trajectory).to_csv(filePathIndividual, header=False, index=False)