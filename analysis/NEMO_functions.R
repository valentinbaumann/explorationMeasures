# Functions for NEMO 2022 analysis


# default colors 
default_colors = c("lightsteelblue1", "dodgerblue1", "royalblue2", "royalblue4")
grey_colors = c("grey", "grey40", "grey20")
world_colors = c("#33CC33","#CC3399")



# histogram
createHist = function(data, xVar, bins = NULL, binwidth = NULL, groupVar = NULL){
  
    ggplot(data, aes_string(x = xVar, fill = groupVar))+
      geom_histogram(bins = bins, binwidth = binwidth) 
  
}


# Barplot
createBarplotMeans = function(data, xVar, yVar, groupVar= NULL, facetVar = NULL, facetCols = NULL, color_values = default_colors){
  
  ggplot(data, aes_string(x = xVar, y = yVar, fill = groupVar)) +
    stat_summary(fun = mean, geom = "bar", position = position_dodge(width = 1), alpha = 0.6) +
    stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = 0.5, position = position_dodge(width = 1)) +
    scale_fill_manual(values = color_values)
  
  
}



# Barplot
createBarplotCounts = function(data, xVar, groupVar= NULL, facetVar = NULL, facetCols = NULL, color_values = default_colors){
  
  ggplot(data, aes_string(x = xVar, fill = groupVar)) +
    geom_bar(position="dodge") +
    scale_fill_manual(values = color_values)
  
}


# Violin Plot
createViolin = function(data, xVar, yVar, groupVar = "noGrouping", facetVar = "noFacet", facetCols = NULL, color_values = default_colors, title = NULL){
  
  if (groupVar == "noGrouping"){
    if (facetVar == "noFacet"){   
      ggplot(data, aes_string(x = xVar, y = yVar, fill = xVar)) +
        geom_violin(scale = "count", alpha = 0.6) + 
        stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = 0.5) +
        stat_summary(fun = mean, geom = "point") +
        scale_fill_manual(values = color_values) +
        ggtitle(title)
      
      
    }else{
      ggplot(data, aes_string(x = xVar, y = yVar, fill = xVar)) +
        geom_violin(scale = "count", alpha = 0.6) + 
        stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = 0.5) +
        stat_summary(fun = mean, geom = "point") +
        scale_fill_manual(values = color_values) +
        facet_wrap(facetVar, ncol = facetCols) +
        ggtitle(title)
    }
  }else{
    if (facetVar == "noFacet"){  
      ggplot(data, aes_string(x = xVar, y = yVar, fill = groupVar)) +
        geom_violin(scale = "count", position = position_dodge(width = 1), alpha = 0.6) + 
        stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = 0.5, position = position_dodge(width = 1)) +
        stat_summary(fun = mean, geom = "point", position = position_dodge(width = 1)) +
        scale_fill_manual(values = color_values) +
        ggtitle(title)
      
    }else{
      ggplot(data, aes_string(x = xVar, y = yVar, fill = groupVar)) +
        geom_violin(scale = "count", position = position_dodge(width = 1), alpha = 0.6) + 
        stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = 0.5, position = position_dodge(width = 1)) +
        stat_summary(fun = mean, geom = "point", position = position_dodge(width = 1)) +
        scale_fill_manual(values = color_values) +
        facet_wrap(facetVar, ncol = facetCols) +
        ggtitle(title)
    }
  }
}



# Boxplot
createBoxPlot = function(data, xVar, yVar, groupVar = "noGrouping", facetVar = "noFacet", facetCols = NULL, color_values = default_colors){
  
  if (groupVar == "noGrouping"){
    if (facetVar == "noFacet"){   
      ggplot(data, aes_string(x = xVar, y = yVar, fill = xVar)) +
        geom_boxplot(alpha = 0.6) + 
        scale_fill_manual(values = color_values)
      
      
    }else{
      ggplot(data, aes_string(x = xVar, y = yVar, fill = xVar)) +
        geom_boxplot(alpha = 0.6) + 
        scale_fill_manual(values = color_values) +
        facet_wrap(facetVar, ncol = facetCols)
    }
  }else{
    if (facetVar == "noFacet"){  
      ggplot(data, aes_string(x = xVar, y = yVar, fill = groupVar)) +
        geom_boxplot(position = position_dodge(width = 1), alpha = 0.6) + 
        scale_fill_manual(values = color_values)
      
    }else{
      ggplot(data, aes_string(x = xVar, y = yVar, fill = groupVar)) +
        geom_boxplot(position = position_dodge(width = 1), alpha = 0.6) + 
        scale_fill_manual(values = color_values) +
        facet_wrap(facetVar, ncol = facetCols)
    }
  }
}





# Scatterplot
createScatter = function(data, xVar, yVar, groupVar = NULL, smoothMethod = "loess", modelFormula = "y~x", facetVar = NULL, facetCols = NULL){
  
    ggplot(data, aes_string(x = xVar, y = yVar, color = groupVar)) +
    geom_beeswarm(alpha = 0.6) + 
    geom_smooth(method = smoothMethod, formula = modelFormula) +
    facet_wrap(facetVar, ncol = facetCols)
}








# min max normalization 
normalizeMinMax <- function(x) {
  (x - min(x, na.rm = T)) / (max(x, na.rm = T) - min(x, na.rm = T))
}


# z standardization
zStandardize = function(x) {
  (x - mean(x, na.rm = T)) / (sd(x, na.rm = T))
}


# invert Variable
invert = function(x){
  ((max(x, na.rm = T) - x) + 1)
}
