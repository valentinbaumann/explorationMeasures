# explorationMeasures
data and code for "Towards a characterization of human spatial exploration behavior"


## repository structure
- *position_data* contains the raw trajectories (as .csv files) for each subject, indexed by subjuect identifier and exposition (i.e., *500_1.csv* represents the first exploration round for subject 500). Note that we here only use data from the first exploration round. All trajectories were recorded in three dimensions (x,y,z).Note that trajectories do not contain timestamps (see publication for more details and how trajectories were resampled to a common sampling rate).
- *person_data* contains a SPSS dataset (*person_data.sav*) with the corresponding subject information (age, sex, envrionment type, novelty seeking score)
- *object_data* contains two .csv files with object coordinates for each of the two virtual environments ("green" or "pink") 
- *feature_extraction* contains the Python code used for the extraction of exploration measures as well as for all plots showing raw trajectories
- *position_data_rediscretized* contains numpy files with all traqjctories rediscretized to a regular step length
- *position_data_flightScaled* contains numpy files with all trajectories resampled to the flight scale
- *analysis* contains the R code used for hierarchical clustering and statistical models and all plots related to the analysis
- *output_data* contains output data from feature extraction and plotting

CAVE: note that the code uses relative paths to access the different directories, which may only work on Windows machines. If any errors occur, please specify direct paths.


## feature extraction (Python)
- *main.py* contains the main script, *readData.py* all functions for reading the trajectories and subject dataframe, *staticticalMeasures.py* contains the functions to compute the different measures, *visuals.py* contains plotting functions
- input data: trajectories are read from .csv files and stored in a list of numpy arrays (*path_data*). The dataframe containing is read from a .sav (SPSS) file and stored as a Pandas dataframe (*person_data*)
- output data: all measures are added to the *person_data* table, which is printed as .csv table to the *output_data* folder


### exploration measures in *statisticalMeasures.py*
All Exploration measures can be applied either to a single trajectory (indicated by the key word *compute*: i.e., *computePathLength()*) as well as to a group of trajectories (indicated by the key word *get*: i.e., *getPathLength()*)

Functions are available for the following measures:
- PathLength
- Pausing
- AreaCovered
- Roaming Entropy
- Sinuosity
- Fractal Dimension
- Revisiting
- Turnarounds
- Flight Turnarounds
- Object Visits
- Object Revisits
- Exploration Efficiency 
- Object Efficiency


## analysis (R)
- includes checks for subject exclusion 
- includes hierarchical clustering analysis
- includes novelty seeing analysis

## Contact
For further information please contact M.Sc.Psych. Valentin Baumann
valentin(dot)baumann(at)med(dot)ovgu(dot)de or valentin(dot)adrian(dot)baumann(at)gmail(dot)com
