import os
import numpy as np
import scipy.io as sio
import sys
sys.path.append('./self_py_fun')
from HW8Fun import produce_trunc_mean_cov, plot_trunc_mean, plot_trunc_cov

#global constants
bp_low = 0.5
bp_upp = 6
E_val = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8','CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
subject_name = 'K114'
session_name = '001_BCI_TRN'

#load MATLAB file
mat_path = 'data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat'
eeg_trunc_obj = sio.loadmat(mat_path)
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type   = eeg_trunc_obj['Type']

#make K114 directory
save_directory = subject_name
os.makedirs(save_directory, exist_ok=True)
print(f"Output dir: {save_directory}")

#means & covariances
(signal_tar_mean, signal_ntar_mean,
 signal_tar_cov, signal_ntar_cov, signal_all_cov) = produce_trunc_mean_cov(eeg_trunc_signal, eeg_trunc_type, E_val)

#time axes
length_per_electrode = signal_tar_mean.shape[1]
time_index_mean = np.linspace(0, 800, length_per_electrode)
time_index_cov  = np.arange(length_per_electrode)

#plot & save 
plot_trunc_mean(
    eeg_tar_mean = signal_tar_mean,
    eeg_ntar_mean = signal_ntar_mean,
    subject_name = f"Subject {subject_name}",
    time_index = time_index_mean,
    E_val = E_val,
    electrode_name_ls = electrode_name_ls,
    save_dir = save_directory)

plot_trunc_cov(signal_tar_cov,  "Target",     time_index_cov, f"Subject {subject_name}", E_val, electrode_name_ls, save_directory)
plot_trunc_cov(signal_ntar_cov, "Non-Target", time_index_cov, f"Subject {subject_name}", E_val, electrode_name_ls, save_directory)
plot_trunc_cov(signal_all_cov,  "All",        time_index_cov, f"Subject {subject_name}", E_val, electrode_name_ls, save_directory)

print("Saved figures:",
      f"{save_directory}/Mean.png",
      f"{save_directory}/Covariance_Target.png",
      f"{save_directory}/Covariance_Non-Target.png",
      f"{save_directory}/Covariance_All.png", sep="\n")
