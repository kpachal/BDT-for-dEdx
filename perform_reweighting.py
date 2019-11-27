#!/usr/bin/env python3

import numpy as np
from numpy.lib.recfunctions import append_fields
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy import ndimage
import pickle
import glob

from pandas import DataFrame 

from ROOT import RDataFrame as RDF
from ROOT import TFile

import sys
sys.path.insert(0, 'hep_ml')
sys.path.insert(0, '../hep_ml')
from hep_ml import reweight

import yaml
from optparse import OptionParser

import root_numpy
from ROOT import TParameter, TFile

print("Collecting data")

analysis = "Validation"
regions = ["Background_dEdxControl","Background_pControl","SignalRegion"]
tree_path = "/afs/cern.ch/work/k/kpachal/PixeldEdX/BackgroundEstimate/run/trees/"
output_filename = "BDT_hists.root"

# Columns we want
columns_load = ['dEdx', 'eta', 'p', 'track_met_dPhi', 'met',
           'muonStatus', 'nIBLOverflows', 'nUsedHits']

# Load data
tree_dict = {}
for region in regions : 

  files = glob.glob(tree_path+"/"+analysis+"_"+region+"*")
  #print("In region", region,"found files",files)

  # Load them...
  this_data = root_numpy.root2array(files,
                          treename="tree", branches=columns_load)

  # Add a column that is just 1's, which is where the
  # learned weights will go
  this_data = append_fields(this_data, 'lowToHighMET_BDT_weight', np.ones(len(this_data)), usemask=False)

  # If this is the dEdx control region we want to subdivide it
  # into two based on dEdx values. Do that here:
  if "Background_dEdxControl" in region :
    low_dEdx = this_data[this_data['dEdx'] < 1.8]
    high_dEdx = this_data[this_data['dEdx'] > 1.8]
    tree_dict["low_met_low_dEdx"] = low_dEdx
    tree_dict["low_met_high_dEdx"] = high_dEdx
  elif "Background_pControl" in region :
    tree_dict["high_met_low_dEdx"] = this_data
  else :
    tree_dict["high_met_high_dEdx"] = this_data

# Train reweighting with low dEdx regions.
# As a first pass, use all columns except the two which make up my plane.
print("Beginning training...")

rw_columns = ['eta', 'p', 'track_met_dPhi', 'muonStatus', 'nIBLOverflows', 'nUsedHits']


# Setting up the BDT
# using the default settings from HH4b.
param_list = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_leaf', 'gb_args']
BDT_params = {}
BDT_params['n_estimators'] = 50
BDT_params['learning_rate'] = 0.1
BDT_params['max_depth'] = 3
BDT_params['min_samples_leaf'] = 125
BDT_params['gb_args'] = {'subsample': 0.4}

print("Setting up BDT with parameters:")
for key in param_list:
  print(key+':', BDT_params[key])

print("About to train...")
print("Training on columns:", rw_columns)

original = DataFrame(tree_dict["low_met_low_dEdx"][rw_columns])
target = DataFrame(tree_dict["high_met_low_dEdx"][rw_columns])

myBDT = reweight.GBReweighter(n_estimators=BDT_params['n_estimators'], learning_rate=BDT_params['learning_rate'], 
                              max_depth=BDT_params['max_depth'], 
                              min_samples_leaf=BDT_params['min_samples_leaf'], 
                              gb_args=BDT_params['gb_args'])

original_weights = np.ones(len(tree_dict["low_met_low_dEdx"]))
target_weights = np.ones(len(tree_dict["high_met_low_dEdx"]))

# Do the fit
myBDT.fit(original, target, original_weight=original_weights, target_weight=target_weights)
print("Training complete!")

print("Predicting weights for high-dEdx, low-MET events...")
# This now predicts weights for my high-dEdx events:
tree_dict["low_met_high_dEdx"]["lowToHighMET_BDT_weight"] = myBDT.predict_weights(DataFrame(tree_dict["low_met_high_dEdx"][rw_columns]))
# And for normalisation reasons, do the same to my low-dEdx events:
tree_dict["low_met_low_dEdx"]["lowToHighMET_BDT_weight"] = myBDT.predict_weights(DataFrame(tree_dict["low_met_low_dEdx"][rw_columns]))

# Number of events at high MET is just event count, as they are not weighted.
print("Number of events in high-MET, low-dEdx region:",len(tree_dict["high_met_low_dEdx"]))
print("Number of events in signal region:",len(tree_dict["high_met_high_dEdx"]))
# Check normalisation: take sum of weights columns after learning.
print("Number of events, before reweighting, in low-MET, low-dEdx region:",len(tree_dict["low_met_low_dEdx"]))
print("Number of events, before reweighting, in low-MET, high-dEdx region:",len(tree_dict["low_met_high_dEdx"]))
# After weighting, we have re-shaped the event distribution but not normalised it.
# Start by seeing what the weighted number of events is.
after_weighting = np.sum(tree_dict["low_met_low_dEdx"]["lowToHighMET_BDT_weight"])
print("Sum of weights, low-MET and low-dEdx:",after_weighting)
# Normalisation factor is the desired number of events over this value.
norm_factor = len(tree_dict["high_met_low_dEdx"])/after_weighting
print("Normalisation factor is:",norm_factor)
# Now in the signal region, our predicted number of events is the normalisation factor
# multiplied by the sum of weights of our reweighted low-MET region.
after_weighting_highdEdx = np.sum(tree_dict["low_met_high_dEdx"]["lowToHighMET_BDT_weight"])
print("Sum of weights, low-MET and high-dEdx:",after_weighting_highdEdx)
signal_pred = norm_factor*after_weighting_highdEdx
print("Predicted number of signal events is:",signal_pred)

# # Now try learning a reweighting from low to high dEdx.
# target = DataFrame(tree_dict["low_met_high_dEdx"][rw_columns])
# target_weights = np.ones(len(tree_dict["low_met_high_dEdx"]))
# myBDT.fit(original, target, original_weight=original_weights, target_weight=target_weights)
# print("Second training complete!")

# Make some histograms to look at how this performed.
hist_dict = {}
for tree in tree_dict.keys() :
  frame = RDF(root_numpy.array2tree(tree_dict[tree]))

  # Hists
  dEdx = frame.Histo1D((tree+"_dEdx","dEdx",100,0,6),"dEdx")
  dEdx_weighted = frame.Histo1D((tree+"_dEdx_weighted","dEdx_weighted",100,0,6),"dEdx","lowToHighMET_BDT_weight")
  p = frame.Histo1D((tree+"_p","p",200, -2500, 2500), "p")
  p_weighted = frame.Histo1D((tree+"_p_weighted","p",200, -2500, 2500), "p", "lowToHighMET_BDT_weight")
  eta = frame.Histo1D((tree+"_eta","eta",60, -3,3),"eta")
  eta_weighted = frame.Histo1D((tree+"_eta_weighted","eta",60, -3,3),"eta", "lowToHighMET_BDT_weight")
  met = frame.Histo1D((tree+"_met","met",200, 0, 1000),"met")
  met_weighted = frame.Histo1D((tree+"_met_weighted","met",200, 0, 1000),"met", "lowToHighMET_BDT_weight")  
  dPhi = frame.Histo1D((tree+"_dPhi","dPhi",64, -3.2, 3.2),"track_met_dPhi")
  dPhi_weighted = frame.Histo1D((tree+"_dPhi_weighted","dPhi",64, -3.2, 3.2),"track_met_dPhi", "lowToHighMET_BDT_weight")
  muonStatus = frame.Histo1D((tree+"_muonStatus","muonStatus",2, -0.5, 1.5),"muonStatus")
  muonStatus_weighted = frame.Histo1D((tree+"_muonStatus_weighted","muonStatus",2, -0.5, 1.5),"muonStatus", "lowToHighMET_BDT_weight")
  nIBLOverflows = frame.Histo1D((tree+"_nIBLOverflows","nIBLOverflows",2, -0.5, 1.5),"nIBLOverflows")
  nIBLOverflows_weighted = frame.Histo1D((tree+"_nIBLOverflows_weighted","nIBLOverflows",2, -0.5, 1.5),"nIBLOverflows","lowToHighMET_BDT_weight")
  nUsedHits = frame.Histo1D((tree+"_nUsedHits","nUsedHits",10, -0.5, 9.5),"nUsedHits")
  nUsedHits_weighted = frame.Histo1D((tree+"_nUsedHits_weighted","nUsedHits",10, -0.5, 9.5),"nUsedHits","lowToHighMET_BDT_weight")

  # Save
  histlist = [dEdx,dEdx_weighted,p,p_weighted,eta,eta_weighted,met,met_weighted,dPhi,dPhi_weighted,muonStatus,muonStatus_weighted,nIBLOverflows,nIBLOverflows_weighted,nUsedHits,nUsedHits_weighted]
  hist_dict[tree] = histlist

print("Saving histograms to file",output_filename)
output_file = TFile.Open(output_filename,"RECREATE")
output_file.cd()
for tree in hist_dict.keys() :
  for hist in hist_dict[tree] :
    hist.Write()
output_file.Close()


