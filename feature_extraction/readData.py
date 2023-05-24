
##### all functions related to loading the raw data

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


def loadPersonData(pathToMainTable):
    """Loads the dataset with all the information associated with the idnividual trajectories. Requires an .sav (SPSS) file as input."""    
    df_persons = pd.read_spss(os.path.join(pathToMainTable, 'person_data.sav'))
    return df_persons



def loadLogsFromCSV(path): 
    
    fileList = os.listdir(path) # get file list
    os.chdir(path)
    res = []
    
    print("reading files from: " + path)

    for p in fileList:
        mytraj_df = pd.read_csv(p, sep=",", header=None)

        mytraj = mytraj_df.values

        res.append(mytraj)
        print(p)
    return res


    