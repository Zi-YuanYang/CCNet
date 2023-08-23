import numpy as np
import pickle
import os
import shutil

import sys
import scipy.io as io
import h5py

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# python getGI.py  path/to/scores.txt  rst_folder

if len(sys.argv) < 3:
    print('getEER.py: input args error! using default ...')
    pathScore = './scores.txt' 
    surname = 'scores'
else:
    pathScore = sys.argv[1]    
    surname = sys.argv[2]

pathIn = os.path.dirname(pathScore)
scorename = os.path.basename(pathScore)
    

#print(sys.argv)
print('\n')
print('pathIn: ', pathIn)
print('scorename: ', scorename)
print('surname:', surname)

print('start to load matching scores ...\n')



pathOut = os.path.join(pathIn, surname)
if os.path.exists(pathOut)==False:
    os.makedirs(pathOut)


# From .pkl:

# pathInner  = os.path.join(pathIn, 'innerScore.pkl')
# pathOuter = os.path.join(pathIn, 'outerScore.pkl')

# pklfile = open(pathInner, 'rb')
# inner = pickle.load(pklfile, encoding='iso-8859-1')
# pklfile.close()

# pklfile = open(pathOuter, 'rb')
# outer = pickle.load(pklfile, encoding='iso-8859-1')
# pklfile.close()

# From .txt:
scores = np.loadtxt(pathScore)

# From old .mat:
# data = io.loadmat(pathScore)
# scores = data['rsts']

# From big .mat -v7:
# scores = h5py.File(pathScore)
# scores = scores['rsts']
# scores = np.transpose(scores)

# print(scores)

# genuine label == 1, impostor label == -1
# scores[matching score, label]
inscore = scores[scores[:, 1]==1, 0]
outscore = scores[scores[:,1]==-1, 0]


# print(inscore)
# print(outscore)
print('inner  (min, max, mean, std): [%f, %f] [%f +- %f]'%(inscore.min(), inscore.max(), inscore.mean(), inscore.std()))
print('outer (min, max, mean, std): [%f, %f] [%f +- %f]'%(outscore.min(), outscore.max(), outscore.mean(), outscore.std()))

print('scores loading done! start to plot histograms ...')


maxvin = np.max(inscore)
minvin = np.min(inscore)
# print(maxvin)
maxvo = np.max(outscore)
minvo = np.min(outscore)
# print(maxvo)


meanvin = np.mean(inscore)
stdvin = np.std(inscore)

meanvo = np.mean(outscore)
stdvo = np.std(outscore)


samples = 100


inscore = (inscore-minvin)/(maxvin-minvin)*samples  # 0~samples
outscore =(outscore-minvo)/(maxvo-minvo)*samples    # 0~samples


histin = np.zeros((samples+1, 1), dtype='int32')
histo = np.zeros((samples+1, 1), dtype='int32')


histin = histin[:,0]
histo = histo[:,0]

# 0-100
for i in inscore:    
    i = int(round(i))
    histin[i] += 1
for i in outscore:
    i = int(round(i))
    histo[i] += 1

histin = np.array(histin)
histo = np.array(histo)


sumtmp = np.sum(histin)
histin = histin / sumtmp * 100

sumtmp = np.sum(histo)
histo = histo / sumtmp * 100

plt.figure(1)

plt.plot(np.linspace(0,1,samples+1)*(maxvo-minvo)+minvo, histo, 'r', label='Impostor')
plt.plot(np.linspace(0,1,samples+1)*(maxvin-minvin)+minvin, histin, 'b', label='Genuine')

plt.legend(loc='upper right', fontsize=13)
plt.xlabel('Matching Score', fontsize=13)
plt.ylabel('Percentage (%)', fontsize=13)
# plt.xscale('log')
plt.ylim([0, 1.2*np.max([histin.max(), histo.max()])])
plt.grid(True)

plt.savefig(os.path.join(pathOut, 'GI_curve.png'))
# plt.show()



with open(os.path.join(pathOut, 'matching_score_distr.txt'), 'w') as f:
    f.writelines('[min, max] [mean +- std]\n')
    f.writelines('inner: [%.10f, %.10f] [%.10f +- %.10f]\n'%(minvin, maxvin, meanvin, stdvin))
    f.writelines('outer: [%.10f, %.10f] [%.10f +- %.10f]\n'%(minvo, maxvo, meanvo, stdvo))
    f.writelines('number of genuine matching:  %d\n'%inscore.shape)
    f.writelines('number of impostor matching: %d\n'%outscore.shape)

    
xin = np.linspace(0,1,samples+1)*(maxvin-minvin)+minvin
xo = np.linspace(0,1,samples+1)*(maxvo-minvo)+minvo

with open(os.path.join(pathOut, 'matching_hist.txt'), 'w') as f:
    for i in range(samples+1):
        f.writelines('%.4f %.4f %.4f %.4f\n'%(xin[i], histin[i], xo[i], histo[i]))

print('done!\n')
