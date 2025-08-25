# Copyright (c) 2023 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: FLAD, Adaptive Federated Learning for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import h5py
import glob
import torch
import scipy.stats
from collections import OrderedDict

from scipy.stats import wasserstein_distance
from sklearn.utils import shuffle
import random as rn

# import tenseal as ts  # Temporarily commented out to avoid mutex lock issue
from imblearn.over_sampling import SMOTE


def apply_smote(X, Y):
    """
    Apply SMOTE to balance class distribution in each client dataset.
    - X: Features (must be 2D for SMOTE)
    - Y: Labels
    """
    original_shape = X.shape  # Store original shape for reshaping later

    # Flatten input if it's not 2D
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)

    # Reshape back to original dimensions
    if len(original_shape) > 2:
        X_resampled = X_resampled.reshape(-1, *original_shape[1:])

    return X_resampled, Y_resampled

def load_dataset(path):
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])  # features
    set_y_orig = np.array(dataset["set_y"][:])  # labels

    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig#.reshape((1, set_y_orig.shape[0]))

    return X_train, Y_train
'''
def load_dataset(path):
    """Load dataset from HDF5 file."""
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    X = np.array(dataset["set_x"][:])
    Y = np.array(dataset["set_y"][:])
    return X, Y
'''

def load_set(folder_path, set_type, seed):
    set_list = []
    time_window = 0
    max_flow_len = 0
    dataset_name = 0
    subfolders = glob.glob(folder_path + "/*/")
    if len(subfolders) == 0:  # for the case in which the is only one folder, and this folder is args.train[0]
        subfolders = [folder_path + "/"]
    else:
        subfolders = sorted(subfolders)
    for dataset_folder in subfolders:
        dataset_folder = dataset_folder.replace("//", "/")  # remove double slashes when needed
        files = glob.glob(dataset_folder + "/*" + '-' + set_type + '.hdf5')

        for file in files:
            filename = file.split('/')[-1].strip()
            tw = int(filename.split('-')[0].strip().replace('t', ''))
            mfl = int(filename.split('-')[1].strip().replace('n', ''))
            dn = filename.split('-')[2].strip()
            if time_window == 0:
                time_window = tw
            else:
                if tw != time_window:
                    print("Mismatching time window size among datasets!")
                    return None
            if max_flow_len == 0:
                max_flow_len = mfl
            else:
                if mfl != max_flow_len:
                    print("Mismatching flow length size among datasets!")
                    return None
            if dataset_name == 0:
                dataset_name = dn
            else:
                if dn != dataset_name:
                    print("Mismatching dataset type among datasets!")
                    return None

            set_list.append(load_dataset(file))

    # Concatenation of all the training and validation sets
    X = set_list[0][0]
    Y = set_list[0][1]
    for n in range(1, len(set_list)):
        X = np.concatenate((X, set_list[n][0]), axis=0)
        Y = np.concatenate((Y, set_list[n][1]), axis=0)

    X, Y = shuffle(X, Y, random_state=seed)

    # 🔥 Apply SMOTE before returning dataset
    #print("Applying SMOTE to balance dataset...")
    #X, Y = apply_smote(X, Y)

    return X, Y, time_window, max_flow_len, dataset_name

'''
def load_set(folder_path, set_type, seed):
    """Load training and validation sets."""
    files = glob.glob(folder_path + f"/*-{set_type}.hdf5")
    X, Y = load_dataset(files[0])

    for file in files[1:]:
        X_new, Y_new = load_dataset(file)
        X = np.concatenate((X, X_new), axis=0)
        Y = np.concatenate((Y, Y_new), axis=0)

    return X, Y, 0, 0, 0  # Dummy values for time_window, max_flow_len, dataset_name
'''
def scale_linear_bycolumn(rawpoints, mins,maxs,high=1.0, low=0.0):
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def create_ckks_context(scale=2**30):
    """Create a CKKS encryption context."""
    # Temporarily disabled due to TenSEAL issues
    # context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    # context.global_scale = scale
    # context.generate_galois_keys()
    # return context
    return None


def encrypt_weights(context, weights, chunk_size=256, scale_factor=10 ** 8):
    """
    Encrypt model weights using CKKS.

    Fix:
    - **Increased scale factor** for better precision.
    - **Increased chunk size** to **reduce precision errors**.
    """
    # Temporarily disabled due to TenSEAL issues
    # encrypted_weights = []
    # for weight in weights:
    #     # Flatten weights and apply scaling
    #     flattened = (weight.flatten() * scale_factor).tolist()

    #     for i in range(0, len(flattened), chunk_size):
    #         chunk = flattened[i:i + chunk_size]
    #         encrypted_weights.append(ts.ckks_vector(context, chunk))

    # return encrypted_weights
    return weights  # Return unencrypted weights for now

def decrypt_weights(context, encrypted_weights, original_shapes, scale_factor=10 ** 8):
    """
    Decrypts encrypted weights using CKKS and reshapes them.

    Fix:
    - Increased scale factor to **reduce numerical loss**.
    - Ensure decrypted weights are **not NaN or Inf**.
    """
    # Temporarily disabled due to TenSEAL issues
    # decrypted_weights = []
    # idx = 0

    # for shape in original_shapes:
    #     size = np.prod(shape)  # Total number of elements in the layer
    #     decrypted_data = []

    #     while len(decrypted_data) < size and idx < len(encrypted_weights):
    #         decrypted_chunk = encrypted_weights[idx].decrypt()
    #         decrypted_data.extend(decrypted_chunk)
    #         idx += 1

    #     # Ensure the correct number of elements
    #     if len(decrypted_data) != size:
    #         print(f"Warning: Expected {size} elements but got {len(decrypted_data)}.")
    #         continue

    #     # Convert to tensor, reshape, and normalize
    #     reshaped_tensor = torch.tensor(decrypted_data, dtype=torch.float32).reshape(shape) / scale_factor

    #     # Check for NaNs
    #     if torch.isnan(reshaped_tensor).any() or torch.isinf(reshaped_tensor).any():
    #         print("Warning: NaN/Inf values detected in decrypted weights! Resetting layer.")
    #         reshaped_tensor = torch.zeros(shape, dtype=torch.float32)

    #     decrypted_weights.append(reshaped_tensor.numpy())

    # return decrypted_weights
    return encrypted_weights  # Return as-is for now

def aggregate_encrypted_weights(encrypted_weights_list):
    """Aggregate encrypted weights from multiple clients."""
    # Temporarily disabled due to TenSEAL issues
    # aggregated_weights = encrypted_weights_list[0]
    # for weights in encrypted_weights_list[1:]:
    #     aggregated_weights = [w1 + w2 for w1, w2 in zip(aggregated_weights, weights)]
    # return aggregated_weights
    
    # Simple aggregation without encryption
    if len(encrypted_weights_list) == 0:
        return []
    
    aggregated_weights = encrypted_weights_list[0]
    for weights in encrypted_weights_list[1:]:
        aggregated_weights = [w1 + w2 for w1, w2 in zip(aggregated_weights, weights)]
    return aggregated_weights


def compute_label_distribution(labels, num_classes):
    distribution = np.zeros(num_classes)
    unique, counts = np.unique(labels.astype(int), return_counts=True)  # Force integer labels
    for u, c in zip(unique, counts):
        distribution[u] = c
    distribution /= len(labels)
    return distribution

def compute_emd(dist1, dist2):
    """
    Compute Earth Mover's Distance (EMD) between two distributions.
    """
    return wasserstein_distance(dist1, dist2)

def select_clients_using_emd(clients, num_selected, num_classes):
    client_distributions = []
    for i, client in enumerate(clients):
        dist = compute_label_distribution(client['training'][1], num_classes)
        client_distributions.append(dist)
        print(f"Client {i} ({client['name']}) Distribution: {dist}")  # <-- ADD THIS

    global_dist = np.mean(client_distributions, axis=0)
    print(f"\nGlobal Distribution: {global_dist}\n")  # <-- ADD THIS

    emd_scores = []
    for i, dist in enumerate(client_distributions):
        emd = compute_emd(dist, global_dist)
        emd_scores.append(emd)
        print(f"Client {i} EMD: {emd:.4f}")  # <-- ADD THIS

    selected_indices = np.argsort(emd_scores)[:num_selected]
    # Print client distributions and EMD scores
    print("\n=== Client Label Distributions ===")
    for i, client in enumerate(clients):
        dist = compute_label_distribution(client['training'][1], num_classes)
        print(f"Client {i} ({client['name']}): {np.round(dist, 2)}")

    global_dist = np.mean(client_distributions, axis=0)
    print(f"\nGlobal Distribution: {np.round(global_dist, 2)}")

    print("\n=== EMD Scores ===")
    for i, score in enumerate(emd_scores):
        print(f"Client {i}: {score:.4f}")

    selected_names = [clients[i]['name'] for i in selected_indices]
    print(f"\nSelected Clients: {selected_names}\n")
    return [clients[i] for i in selected_indices]