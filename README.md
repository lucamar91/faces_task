# faces_task

# 25x30, 50x60
These are the folders containing the inputs of the scripts, i.e. the databases of geometric, texture (uniformed), concatenated and eigenface coordinates for the FEI face database, collected in .dat files.

# classifier-error_bars.py
This script performs a nearest-neighbour binary classification task, based on the Mahalanobis distance, and returns a .dat file with the success rates of the classification task and the corresponding error bars for all possible numbers of principal components + a figure for a quick visualization of these same results.
The first three string parameters in the script, i.e. (1) coord_type, (2) dimension, (3) criterion determine respectively (1) the type of coordinates that are fed to the classification algorithm (chosen among the 4 possibilities listed above), (2) the size of the pictures from which this coordinates are obtained, (3) the specific task that is performed, chosen between a gender (male/female) classification and an expression (neutral/smiling) classification. 

# recognize-a-face.py
This script performs facial recognition, i.e. given a smiling face tries to identify the picture in the database that portraits the same person but with a neutral expression (and viceversa). This is also a classification task, but in this case the classes among which the algorithm has to choose are not two but N (the number of distinct subjects in the FEI database). Returns: again a .dat file and a figure with the success rates and the error bars.
As in the other task, the size of the images and the type of the coordinates can be selected by modifying respectively the int variables w (idth), h (eight) and the string variable coord_type.

# useful_functions_def.py
It contains some functions used in the other scripts to compute the correlation matrices and the Mahalanobis distance.

# ABOUT THE LIBRARIES
Please note that some basic Python libraries are needed to run these scripts, here an exhaustive list:
- numpy
- matplotlib
- os
- scipy
- datetime
- math
- sklearn
- PIL
- time
- sys
