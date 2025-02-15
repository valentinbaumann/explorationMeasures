

 
####################  import libraries #####################
source("explorationMeasures_libraries.R")


#################### source functions  #####################
source("explorationMeasures_functions.R")


#################### read data #####################

setwd('../output_data') # read data from python pipeline output folder
data_NEMO_raw = read.csv("DataExploration_NEMO.csv", header = TRUE)
data_SILCTON_raw = read.csv("DataExploration_SILCTON.csv", header = TRUE)
#str(data_NEMO_raw)

setwd('../analysis') # go back to main analysis directory






#################### fix variables #####################


#### NEMO
data_NEMO = data_NEMO_raw
# factorize variables
data_NEMO$Subject = as.character(data_NEMO$Subject)
data_NEMO$Subject = as.factor(data_NEMO$Subject)
# reorder levels
data_NEMO$VE_sequence = factor(data_NEMO$VE_sequence, levels = c("GreenGreen", "PinkPink", "GreenPink", "PinkGreen"))
# drop unused levels
data_NEMO = droplevels.data.frame(data_NEMO)



#### SILCTON

data_SILCTON = data_SILCTON_raw
# factorize variables
data_SILCTON$Subject = as.character(data_SILCTON$Subject)
data_SILCTON$Subject = as.factor(data_SILCTON$Subject)


data_SILCTON$gender = as.character(data_SILCTON$gender)
data_SILCTON$gender[data_SILCTON$gender == "F"] = "female" 
data_SILCTON$gender[data_SILCTON$gender == "M"] = "male" 
data_SILCTON$gender = as.factor(data_SILCTON$gender)

# drop unused levels
data_SILCTON = droplevels.data.frame(data_SILCTON)





