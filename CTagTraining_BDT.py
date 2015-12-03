from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.max_open_warning"] = -1

# Print options
import numpy as np
np.set_printoptions(precision=3)

import root_numpy as rootnp



# training

variables = [
	'trackSip2dSig_0',
	'trackSip2dSig_1',
	'trackSip3dSig_0',
	'trackSip3dSig_1',
	'trackPtRel_0',
	'trackPtRel_1',
	'trackPPar_0',
	'trackPPar_1',
	'trackPtRatio_0',
	'trackJetDist_0',
	'trackDecayLenVal_0',
	'vertexEnergyRatio_0',
	'trackSip2dSigAboveCharm_0',
	'trackSip3dSigAboveCharm_0',
	'flightDistance2dSig_0',
	'flightDistance3dSig_0',
	'trackSumJetEtRatio',
	'trackSumJetDeltaR',
	'massVertexEnergyFraction_0',
	'leptonRatioRel_0',
]


signal_files = [
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVNoVertex_C.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVPseudoVertex_C.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVRecoVertex_C.root"
    ]
bckgr_files = [  
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVNoVertex_B.root",
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVPseudoVertex_B.root",
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVRecoVertex_B.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVNoVertex_DUSG.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVPseudoVertex_DUSG.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1_WithNewWeights/QCD/skimmed_20k_eachptetabin_CombinedSVRecoVertex_DUSG.root"
    ]



print 'Merging and converting the samples'
nfiles_per_sample = None
skip_n_events = 10
    
#root_numpy.root2array(filenames, treename=None, branches=None, selection=None, start=None, stop=None, step=None, include_weight=False, weight_name='weight', cache_size=-1)
signal_merged = np.ndarray((0,len(variables)),float)
bckgr_merged = np.ndarray((0,len(variables)),float)
for f_sig in signal_files:
	signal = rootnp.root2array(f_sig,'tree',variables,None,0,nfiles_per_sample,skip_n_events,False,'weight')
	signal = rootnp.rec2array(signal)
	signal_merged = np.concatenate((signal_merged,signal),0)
for f_bck in bckgr_files:	
	bckgr = rootnp.root2array(f_bck,'tree',variables,None,0,nfiles_per_sample,skip_n_events,False,'weight')
	bckgr = rootnp.rec2array(bckgr)
	bckgr_merged = np.concatenate((bckgr_merged,bckgr),0)


X = np.concatenate((signal_merged, bckgr_merged))
y = np.concatenate((np.ones(signal_merged.shape[0]),np.zeros(bckgr_merged.shape[0])))


print 'Getting event weights from the trees'
# Get the weights
weights = np.ones(0)
for f_sig in signal_files:
	weights_sig = rootnp.root2array(f_sig,'tree','weight',None,0,nfiles_per_sample,skip_n_events,False,'weight')
	weights = np.concatenate((weights,weights_sig),0)
for f_bck in bckgr_files:	
	weights_bckgr = rootnp.root2array(f_bck,'tree','weight',None,0,nfiles_per_sample,skip_n_events,False,'weight')
	weights = np.concatenate((weights,weights_bckgr),0)


print 'Starting training'
# Do the trainings
import time	
from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators=500,min_samples_split = 50,n_jobs = 5, verbose = 3)
start = time.time()
clf.fit(X, y,weights)
end = time.time()
print 'training completed --> Elapsed time: ' , (end-start)/60 ,  'minutes'






# Validation

val_signal_files = [
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVNoVertex_C.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVPseudoVertex_C.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVRecoVertex_C.root"
    ]
val_bckgr_files = [  
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVNoVertex_B.root",
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVPseudoVertex_B.root",
    #"/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVRecoVertex_B.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVNoVertex_DUSG.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVPseudoVertex_DUSG.root",
    "/user/smoortga/CTag/TMVActag_v1/FlatTrees_SL_7_5_1/TTbar/CombinedSVRecoVertex_DUSG.root"
    ]


print 'Starting validation'

#root_numpy.root2array(filenames, treename=None, branches=None, selection=None, start=None, stop=None, step=None, include_weight=False, weight_name='weight', cache_size=-1)
val_signal_merged = np.ndarray((0,len(variables)),float)
val_bckgr_merged = np.ndarray((0,len(variables)),float)
for f_sig in val_signal_files:
	val_signal = rootnp.root2array(f_sig,'tree',variables,None,0,nfiles_per_sample,skip_n_events,False,'weight')
	val_signal = rootnp.rec2array(val_signal)
	val_signal_merged = np.concatenate((val_signal_merged,val_signal),0)
for f_bck in val_bckgr_files:	
	val_bckgr = rootnp.root2array(f_bck,'tree',variables,None,0,nfiles_per_sample,skip_n_events,False,'weight')
	val_bckgr = rootnp.rec2array(val_bckgr)
	val_bckgr_merged = np.concatenate((val_bckgr_merged,val_bckgr),0)

X_val = np.concatenate((val_signal_merged, val_bckgr_merged))
y_val = np.concatenate((np.ones(val_signal_merged.shape[0]),np.zeros(val_bckgr_merged.shape[0])))

from sklearn.metrics import roc_curve
#fpr, tpr, thresholds = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
fpr, tpr, thresholds = roc_curve(y_val, clf.predict_proba(X_val)[:, 1])
plt.semilogy(tpr, fpr,label='RFC, ntrees = 500')
plt.ylabel("Light Efficiency")
plt.xlabel("Charm Efficiency")
plt.legend(loc='best')
plt.grid(True)
plt.savefig("CvsLight_ROC_test.png")
#plt.show()
