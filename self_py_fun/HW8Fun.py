import numpy as np
import matplotlib.pyplot as plt

def produce_trunc_mean_cov(input_signal, input_type, E_val):
    input_type = np.squeeze(input_type)
    sample_size_len = input_signal.shape[0]
    feature_len = input_signal.shape[1]
    length_per_electrode = feature_len // E_val
    X = input_signal.reshape(sample_size_len, E_val, length_per_electrode)
    X_ntar = X[input_type == -1]
    X_tar = X[input_type == 1]
    signal_ntar_mean = np.mean(X_ntar, axis=0)
    signal_tar_mean = np.mean(X_tar, axis=0)
    signal_tar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_ntar_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    signal_all_cov = np.zeros((E_val, length_per_electrode, length_per_electrode))
    for e in range(E_val):
        signal_tar_cov[e] = np.cov(X_tar[:, e, :], rowvar=False, ddof=1)
        signal_ntar_cov[e] = np.cov(X_ntar[:, e, :], rowvar=False, ddof=1)
        signal_all_cov[e] = np.cov(X[:, e, :], rowvar=False, ddof=1)
    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]

def plot_trunc_mean(eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls, save_dir, y_limit=np.array([-5, 8]), fig_size=(12, 12)):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        row = i % 4
        col = i // 4
        ax = axes[row, col]
        ax.plot(time_index, eeg_tar_mean[i], color='red', label='Target')
        ax.plot(time_index, eeg_ntar_mean[i], color='blue', label='Non-Target')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (muV)')
        ax.set_title(electrode_name_ls[i])
        if i == 0:  
            ax.legend()
    fig.suptitle(subject_name)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Mean.png')
    plt.close()

def plot_trunc_cov(eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls, save_dir, fig_size=(14, 12)):
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    fig.suptitle(f'{cov_type} - Subject: {subject_name}', fontsize=14, weight='bold')
    X, Y = np.meshgrid(time_index, time_index)
    for e in range(E_val):
        r = e % 4  
        c = e // 4
        ax = axes[r, c]
        cont = ax.contourf(X, Y, eeg_cov[e, :, :], cmap='plasma')
        ax.set_title(electrode_name_ls[e])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Time (ms)')
        ax.invert_yaxis()
    fig.colorbar(cont, ax=axes, orientation='vertical', fraction=0.03, pad=0.05)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Covariance_{cov_type}.png')
    plt.close()
