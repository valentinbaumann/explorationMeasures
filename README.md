# ExplorationMeasures
This is the repository for Baumann et al.: "Towards a characterization of human spatial exploration behavior" (unpublished). For background information on the datasets, please see [Schomaker et al., (2022). *Effects of exploring a novel environment on memory across the lifespan*](https://www.nature.com/articles/s41598-022-20562-4) for the NEMO dataset as well as [Brunec et al., (2022). *Exploration patterns shape cognitive map learning*](https://www.sciencedirect.com/science/article/abs/pii/S0010027722003493) for the SILCTON dataset.

![visualization of trajectories](./pictures/img_trajectories.png "visualization of trajectories")

## Repository structure
- *position_data_NEMO* contains the raw trajectories (as .csv files) for each subject in the NEMO dataset. Trajectories are indexed by subject number and the exploration round (i.e., *500_1.csv* represents the trajectory of the first exploration round for subject 500). Note that while the original dataset contained two rounds of exploration, we here only use data from the first exploration round. All trajectories were recorded in three dimensions (order of columns: x,y,z) Note that trajectories do not contain timestamps (see publication for more details and how trajectories were resampled to a common sampling rate).
- *position_data_SILCTON_EXP1* contains the raw trajectories (as .csv files) for each subject of the first experiment in the SILCTON dataset. Trajectories are indexed by the subject number (i.e., *0059_log.csv* represents the trajectory for subject 59). Note that we were not able to obtain logs for the subjects 60, 61, 63, 68, 71, 72, 74, 75, 76, 91, 95, 99, 101, 103, 109, 116, 118, 119, 123, 124, 126, 130, 132, 133, 135, 139 and 140. All trajectories were recorded in three dimensions (order of columns: x,y,z), whith an additionnal fourth column representing heading direction. Note that trajectories do not contain timestamps (according to Brunec et al., we assume a sampling rate of 10Hz). Also, note that in the first experiment, exploration was not continuous, but rather changed between exploration blocks (4 min) and task blocks (1 min).
- *position_data_SILCTON_EXP2* contains the raw trajectories (as .csv files) for each subject of the second experiment in the SILCTON dataset. Trajectories are indexed by the subject number (i.e., *0015.csv* represents the trajectory for subject 15). All trajectories were recorded in three dimensions (order of columns: x,y,z), whith an additionnal fourth column representing heading direction. Note that trajectories do not contain timestamps (according to Brunec et al., we assume a sampling rate of 10Hz).
- *person_data* contains tables (as .csv files) with the corresponding subject information (age, sex, envrionment type...) for the three different datasets (*person_data_NEMO.csv*, *person_data_SILCTON_exp1.csv*, *person_data_SILCTON_exp2.csv*)
- *landmark_data* contains three .csv files with the landmark coordinates for different virtual environments (*objectsNEMO_green.csv* and *objectsNEMO_pink.csv* for the two NEMO environemnts, *objectsSILCTON.csv* for the SILCTON environment). Also includes a map of the SILCTON environment as a .png file.
- *feature_extraction* contains the Python code used for the extraction of exploration measures as well as for all plots showing raw trajectories
- *flight_data_NEMO* contains .csv files with all trajectories in the NEMO dataset resampled to the flight scale following the same indexing order as detailed above(i.e., *500_1_flight.csv* represents the flight scaled trajectory for subject 500). Note that flight scaled trajectories are in 2d (x,z coordinates only).
- *flight_data_SILCTON* contains .csv files with all trajectories in the SILCTON dataset resampled to the flight scale. Trajectories are indexed by experiment number as well as subject number (i.e., *1_59_flight.csv* represents the flight scaled trajectory for subject 59 from the first experiment). Note that flight scaled trajectories are in 2d (x,z coordinates only).
- *analysis* contains the R code used for hierarchical clustering and statistical models and all plots related to the analysis
- *output_data* contains output data from feature extraction and plotting

CAVE: note that the code uses relative paths to access the different directories, which may only work on Windows machines. If any errors occur, please specify direct paths.


## Feature extraction (Python)
- *main.py* contains the main script, *readData.py* all functions for reading the trajectories and subject dataframe, *staticticalMeasures.py* contains the functions to compute the different measures, *visuals.py* contains plotting functions
- input data: trajectories are read from .csv files and stored in a list of numpy arrays (*path_data_NEMO*, *path_data_SILCTON*). The pandas dataframes containing the subject information (*person_data_NEMO*, *person_data_SILCTON*) are read from the corresponding .csv files.
- output data: all measures are added to the respective *person_data* pandas dataframes, which are then saved as .csv files to the *output_data* folder


### Exploration measures in *statisticalMeasures.py*
All Exploration measures can be applied either to a single trajectory (indicated by the key word *compute*: i.e., *computePathLength()*) as well as to a group of trajectories (indicated by the key word *get*: i.e., *getPathLength()*)

Functions are available for the following measures:
- Path Length
- Pausing
- Area Covered
- Roaming Entropy
- Minimum Convex Polygon
- Sinuosity
- Fractal Dimension
- Revisiting
- Turnarounds
- Flight Turnarounds
- Landmark Visits
- Landmark Revisits
- Exploration Efficiency 
- Object Efficiency

Additionally, we also provide functions to assess the data quality:
- percentage of lag steps
- variability of step lengths 

## Analysis (R)
- includes checks for subject exclusions
- includes hierarchical clustering analysis for the NEMO and SILCTON datasets

## Contact
For further information please contact Valentin Baumann
valentin(dot)baumann(at)med(dot)ovgu(dot)de or valentin(dot)adrian(dot)baumann(at)gmail(dot)com

## Other movement analysis packages 
Note that for some measures we adapted or directly call code from two already published software packages, *[trajr]*(https://onlinelibrary.wiley.com/doi/10.1111/eth.12739) and *[traja]*(https://joss.theoj.org/papers/10.21105/joss.03202). If possible, we crossvalidated results between packages to ensure correct computation of features. 