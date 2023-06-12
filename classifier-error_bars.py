import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy.sparse.linalg import eigsh
import math
from useful_functions_def import *
import sys

coord_type = 'conc'                 # geom, eigf, text, conc
dimension = '25x30'                 # 25x30, 50x60
criterion = 'gender'            # classification criterion: gender or expression
print(coord_type, dimension, criterion)
Plim = 80


######################## LOADING DATA ########################
if coord_type == 'geom':
    path = 'geom/'
    label = ''
    neutral = np.loadtxt(path + 'geom_neutral.dat')
    smile = np.loadtxt(path + 'geom_smiling.dat')
elif coord_type == 'conc':
    path = dimension + '/conc/'
    label = '-'+dimension
    neutral = np.loadtxt(path + coord_type + '_5-fold_n0_neutral_' + dimension + '.dat')
    smile = np.loadtxt(path + coord_type + '_5-fold_n0_smiling_' + dimension + '.dat')
else:
    path = dimension + '/' + coord_type + '/'
    label = '-'+dimension
    neutral = np.loadtxt(path + coord_type + '_neutral_' + dimension + '.dat')
    smile = np.loadtxt(path + coord_type + '_smiling_' + dimension + '.dat')

dataset = np.vstack((neutral, smile))

S, D = neutral.shape         
N = D/2
toolbar_width = 50                   ##
notch = 2*S/toolbar_width            ##


######################## DEFINING POSITIVES AND NEGATIVES ########################
male_idx=np.array([1,2,3,4,5,6,7,8,9,10,13,14,16,17,18,19,20,22,23,24,27,30,31,32,33,34,35,36,38,40,41,42,44,45,46,47,
                 48,49,50,52,53,54,55,56,57,59,61,62,63,64,65,66,67,68,70,71,73,74,76,77,80,81,82,83,88,89,90,92,93,
                 94,95,107,108,109,110,111,112,113,114,115,116,118,120,147,148,152,153,154,165,171,182,185,186,187,
                 188,190,191,192,193,194])
female_idx = np.setdiff1d(np.arange(1,201), male_idx)

Pmax = np.amin([D, 2*S-3, Plim])
response = np.zeros((2*S, Pmax))
success = np.zeros((2*S, Pmax))

if criterion == 'gender':
    positive_idx = np.hstack((female_idx -1, female_idx + S - 1))         # positive --> F (indifferent choice)
    negative_idx = np.hstack((male_idx -1, male_idx + S - 1))    
if criterion == 'expression':
    positive_idx = np.arange(200, 400)                                    # positive --> smiling
    negative_idx = np.arange(200)
    
truth = np.zeros((2*S, Pmax))
truth[positive_idx] = 1


sys.stdout.write('\r')                                                           ##
sys.stdout.write("[%s]" % (" " * toolbar_width))                                 ##
sys.stdout.flush()                                                               ##
sys.stdout.write("\b" * (toolbar_width+1))                                       ##


######################## THE CLASSIFICATION CYCLE (LEAVE-ONE-OUT) ########################
for s in range(2*S):
    if s%notch == 0:
        sys.stdout.write("-")
        sys.stdout.flush()

    # masks to select training sets
    negative_tr_idx = np.setdiff1d(negative_idx, s)
    positive_tr_idx = np.setdiff1d(positive_idx, s)
    tr_idx_l1o = np.setdiff1d(np.arange(2*S), s)
    
    # standardisation
    training = dataset[tr_idx_l1o]
    avg = np.average(training, axis=0)
    st_dev = np.std(training, axis=0, ddof=1)
    delta_std = (dataset - avg)/st_dev
        
    # computing the average neutral and smiling faces (already standardised)
    negative_avg = np.average(delta_std[negative_tr_idx], axis=0)
    positive_avg = np.average(delta_std[positive_tr_idx], axis=0)
    
    # PCA
    C_training = correlation_matrix(delta_std[tr_idx_l1o])
    if Pmax == D:
        eigval, eigvec = LA.eigh(C_training)
    else:
        eigval, eigvec = eigsh(C_training, k = Pmax)
    order = np.argsort(eigval)[::-1]
    lambdas = np.copy(eigval)[order]
    E = np.copy(eigvec).T[order]
    
    for p in range(1, Pmax+1):        # column index of 'response' goes from 0 to Pmax-1, but number of PCs goes from 1 to Pmax
        dist_from_negative = mahalanobis_d(delta_std[s], negative_avg, E, lambdas, P = p)
        dist_from_positive = mahalanobis_d(delta_std[s], positive_avg, E, lambdas, P = p)
        # if the classifier returns correct answer success is True:
        response[s, p-1] = (dist_from_positive < dist_from_negative)
        
success = (response == truth)         ######### fuori dal ciclo va bene? sembra di si


sys.stdout.write(']\n')                   ##
    
######################## THE PLOTS ########################
success_rate = np.average(success.astype(float), axis=0) 
error_bars = np.std(success.astype(float), axis=0)/np.sqrt(2*S)

# this are arrays
tp = np.logical_and(response, truth)
tn = np.logical_and(np.logical_not(response), np.logical_not(truth))
fp = np.logical_and(response, np.logical_not(truth))
fn = np.logical_and(np.logical_not(response), truth)
# and this are float values
TP = np.sum(tp, axis=0).astype(float)
TN = np.sum(tn, axis=0).astype(float)
FP = np.sum(fp, axis=0).astype(float)
FN = np.sum(fn, axis=0).astype(float)
# True and False Positive Rate
TPR = TP/(TP+FN)
FPR = FP/(FP+TN)
p_star = np.argmax(TPR-FPR)          # it is the index ( real number of PCs - 1 )

# graph of the success rate
plt.plot(np.arange(1,Pmax+1), success_rate, 'o', markersize=4)
plt.errorbar(np.arange(1,Pmax+1), success_rate, yerr = error_bars, fmt='none')
plt.xlim(0.,Pmax+1)
plt.ylim(0.5,1)
plt.xlabel('$P$' , fontsize = 15)
plt.ylabel('success rate', fontsize = 15)
#plt.axhline(y=0.9, color='orange', linestyle='--')
plt.axvline(x=np.argmax(success_rate)+1, color='orange', linestyle='--')
plt.grid()
plt.savefig(path+coord_type+'-'+criterion+'_classif_w_errorbars'+label+'.png')
plt.clf()

# the ROC curve
plt.plot([0,1], [0,1],'--')
plt.plot(FPR, TPR, 'o', markersize = 7, fillstyle='none', markeredgewidth = 1.)
plt.plot(FPR[p_star], TPR[p_star], 'bo', markersize = 7, fillstyle='none', markeredgewidth = 1.)
plt.annotate(p_star+1, (FPR[p_star], TPR[p_star]), size = 'large', xytext = (FPR[p_star]+.02, TPR[p_star]-.02) )
plt.grid()
plt.ylim(0,1)
plt.xlim(0.,1)
plt.title('$ROC$ curve', fontsize = 18)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
#plt.savefig(path+coord_type+'-'+criterion+'_classif_ROC'+label+'.png')

np.savetxt(path+coord_type+'-'+criterion+'_classif_w_errorbars'+label+'.dat', np.vstack((success_rate, error_bars)) )