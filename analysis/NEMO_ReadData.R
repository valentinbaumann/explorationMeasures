

 
####################  import libraries #####################
source("NEMO_libraries.R")


#################### source functions  #####################
source("NEMO_functions.R")


#################### read data #####################

setwd('../output_data') # read data from python pipeline output folder
raw_data = read.csv("DataExploration.csv", header = TRUE)
#str(raw_data)

setwd('../analysis') # go back to main analysis directory






#################### fix variables #####################

main_data = raw_data
# factorize variables
main_data$Subject = as.character(main_data$Subject)
main_data$Subject = as.factor(main_data$Subject)


# reorder levels
main_data$VE_sequence = factor(main_data$VE_sequence, levels = c("GreenGreen", "PinkPink", "GreenPink", "PinkGreen"))



# compute additional variables
main_data$ObjectRevisits = main_data$ObjectVisits10  - main_data$ObjectsVisited10


# drop unused levels
main_data = droplevels.data.frame(main_data)



