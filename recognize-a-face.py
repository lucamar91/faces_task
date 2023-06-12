import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy.sparse.linalg import eigsh
import os
from PIL import Image
from useful_functions_def import correlation_matrix, standardize, mahalanobis_d, relative_log_dist
import time              ##
import sys               ##
start = time.time()

# - k-fold CV
# - Mah.distance computed using mahalanobis_d in some_other_functions

# SELECT IMAGE SIZE HERE: 25x30, 50x60, 100x120
w=25
h=30
# SELECT COORDINATE TYPE HERE: geom, text, (conc), eigf
coord_type = 'conc'

dimension = '%dx%d' % (w,h)
path = ''
k = 5
Plim = 80        # setting a limit for the number of PCs to be computed

if coord_type == 'geom':
    neutral = np.loadtxt(path + coord_type + '/' + coord_type + '_neutral.dat')
    smiling = np.loadtxt(path + coord_type + '/' + coord_type + '_smiling.dat')    
elif coord_type == 'conc':       # carico i file solo per ricavare S, D; poi dovro caricarli nel ciclo della CV
    neutral = np.loadtxt(path + dimension + '/' + coord_type + '/' + coord_type + '_5-fold_n0_neutral_' + dimension + '.dat')
    smiling = np.loadtxt(path + dimension + '/' + coord_type + '/' + coord_type + '_5-fold_n0_smiling_' + dimension + '.dat')
else:
    neutral = np.loadtxt(path + dimension + '/' + coord_type + '/' + coord_type + '_neutral_' + dimension + '.dat')
    smiling = np.loadtxt(path + dimension + '/' + coord_type + '/' + coord_type + '_smiling_' + dimension + '.dat')
        
print(coord_type)
if coord_type != 'geom':
    print(dimension)
    
dataset = np.vstack((neutral, smiling))
S, D = neutral.shape         # assuming that neutral and smile contain the same subjects


Pmax = np.amin([D, int(S*(k-1)/k), Plim])                # -2 dofs : i could get to P=2*S*(k-1)/k but makes things too long
success = np.zeros((2, Pmax, k))               # will contain the number of successful guesses, for the test and the training
success_log = np.zeros((2, Pmax, k)) 

toolbar_width = 50                   ##
notch = 2*S/toolbar_width            ##


for n in range(k):
    sys.stdout.write('\r')                   ##
    sys.stdout.write("%d/%d [%s]" % (n+1, k, " " * toolbar_width))                   ##
    sys.stdout.flush()                                                               ##
    sys.stdout.write("\b" * (toolbar_width+1))                                       ##

    # selecting the training test
    U = int(float(S)/k)
    
    # excluding S-T neutral and  S-T smiling from the training (in such a way that each face is represented at least once)
    te_idx = np.hstack((np.arange(n*U, (n+1)*U), S + np.arange((n+1)*U, (n+2)*U)%S ))
    tr_idx = np.setdiff1d(np.arange(2*S), te_idx)
    training = dataset[tr_idx]

    # standardization
    avg = np.average(training, axis=0)
    st_dev = np.std(training, axis=0, ddof=1)
    delta_std = (dataset - avg)/st_dev

    # PCA
    C_training = correlation_matrix(training)
    if Pmax == D:
        eigval, eigvec = LA.eigh(C_training)
    else:
        eigval, eigvec = eigsh(C_training, k = Pmax)
    order = np.argsort(eigval)[::-1]
    lambdas = np.copy(eigval)[order]
    E = np.copy(eigvec).T[order]                 # each ROW is a eigvec
    
    #inserire database componenti princpiali
    dataset_pc = delta_std.dot(eigvec)
    
    for face in range(2*S):
        # the progress bar:
        if face%notch == 0:             ##
            sys.stdout.write("-")
            sys.stdout.flush()
            
        for P in range(1, Pmax):                            
            is_in_test = int(face in te_idx)           # 0 for the training, 1 for the test set
            is_not_smiling = int(face < S)                 # 0 for the neutral, 1 for the smiles
                        
            # i want to compare the face only with the faces that belong to the other group (smiling/neutral)
            face_std = delta_std[face]
            the_other_idxs = np.arange(S*is_not_smiling, S*(1+is_not_smiling))
            the_other_faces = delta_std[the_other_idxs]                                  # 2*S faces
            
            # mahal. distance between face and the_other_faces:   
            #mah_d_from_avg = mahalanobis_d(the_other_faces, np.zeros((D,)), E, lambdas, P=P)       ############
            mah_d = mahalanobis_d(face_std, the_other_faces, E, lambdas, P=P)      ###############
            
            #debug
            if mah_d.any() ==0: # or mah_d_from_avg.any() ==0:
                print(face, P)
            
            if face%S == np.argmin(mah_d)%S:
                success[is_in_test, P, n] += 1.                          # the point makes success a float
sys.stdout.write(']\n')                   ##

# obtaining the success rate
success[0] = success[0]/len(tr_idx)
success[1] = success[1]/len(te_idx)
score = np.average(success, axis=2)
score_std = np.std(success, axis=2)/np.sqrt(k)
plt.title('recog ' + dimension + ' ' + coord_type)
plt.plot(score[0], 'ro', ms = 4, label='tr')
plt.plot(score[1], 'go', ms = 4, label='te')
plt.errorbar(np.arange(len(score[0])), score[0], yerr = score_std[0], fmt='none')
plt.errorbar(np.arange(len(score[1])), score[1], yerr = score_std[1], fmt='none')
plt.legend()
plt.grid()

if coord_type == 'geom':
    plt.savefig(coord_type + '/'+coord_type+'-recog_error_bars-%dfoldCV.png' % k)
    np.savetxt(coord_type + '/'+coord_type+'-recog_error_bars-%dfoldCV.dat' % k, np.vstack((score, score_std)))
else:
    plt.savefig(dimension + '/' + coord_type + '/' +coord_type+'-recog_error_bars-%dfoldCV-' % k+dimension+'.png')
    np.savetxt(dimension + '/' + coord_type + '/' +coord_type+'-recog_error_bars-%dfoldCV-' % k+dimension+'.dat', np.vstack((score, score_std)))
        
end = time.time()
duration = end-start
print('Duration: %d minutes and ' % int(duration/60) + str(duration%60) + ' seconds')


