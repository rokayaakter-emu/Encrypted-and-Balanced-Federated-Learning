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
from keras.src.models.cloning import clone_model

from ann_models import *
import time
import math
import csv
from sklearn.metrics import f1_score
import random
import copy
import gc
import shutil

from util_functions import create_ckks_context, encrypt_weights, decrypt_weights, aggregate_encrypted_weights

import csv
import shutil
# import tenseal as ts  # Temporarily commented out to avoid mutex lock issue
from util_functions import encrypt_weights, decrypt_weights, aggregate_encrypted_weights


# General hyperparameters
EXPERIMENTS = 10

# FL hyper-parameters
PATIENCE = 25
CLIENT_FRACTION = 0.8  # Fraction of clients selected at each round for FedAvg-based approaches


def trainClientModel(model, epochs, X_train, Y_train, X_val, Y_val, steps_per_epoch=None):
    if steps_per_epoch != None and steps_per_epoch > 0:
        batch_size = max(int(len(Y_train) / steps_per_epoch), 1)  # min batch size set to 1

    tp0 = time.time()
    history = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size,
                        verbose=2, callbacks=[])
    tp1 = time.time()

    loss_train = history.history['loss'][-1]
    loss_val = history.history['val_loss'][-1]

    return model, loss_train, loss_val, tp1 - tp0

def FederatedTrain(clients, model_type, outdir, time_window, max_flow_len, dataset_name, epochs='auto', steps='auto',
                   training_mode='flad', weighted=False, optimizer='SGD', nr_experiments=EXPERIMENTS, num_classes=2):
    round_fieldnames = ['Model', 'Round', 'AvgF1']
    tuning_fieldnames = ['Model', 'Epochs', 'Steps', 'Mode', 'Weighted', 'Experiment', 'ClientsOrder', 'Round',
                         'TotalClientRounds', 'F1', 'F1_std', 'Time(sec)']

    for client in clients:
        round_fieldnames.append(client['name'] + '(f1)')
        round_fieldnames.append(client['name'] + '(loss)')
        tuning_fieldnames.append(client['name'] + '(f1)')
        tuning_fieldnames.append(client['name'] + '(rounds)')

    hyperparamters_tuning_file = open(outdir + '/federated-tuning.csv', 'w', newline='')
    tuning_writer = csv.DictWriter(hyperparamters_tuning_file, fieldnames=tuning_fieldnames)
    tuning_writer.writeheader()
    hyperparamters_tuning_file.flush()

    # Initialize the server and create CKKS context
    server = init_server(model_type, dataset_name, clients[0]['input_shape'], max_flow_len)
    if server is None:
        exit(-1)

    # Create CKKS Encryption Context
    # server_context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    # server_context.global_scale = 2 ** 40
    # server_context.generate_galois_keys()
    server_context = None  # Temporarily disabled due to TenSEAL issues

    model_name = server['model'].name
    client_indices = list(range(len(clients)))

    training_filename = f"{model_name}-epochs-{epochs}-steps-{steps}-trainingclients-{training_mode}-weighted-{weighted}"
    training_file = open(outdir + '/' + training_filename + '.csv', 'w', newline='')
    writer = csv.DictWriter(training_file, fieldnames=round_fieldnames)
    writer.writeheader()

    best_model = server['model']
    total_rounds = 0
    stop_counter = 0
    max_f1 = 0
    stop = False
    val_losses = []

    while not stop:  # Training epochs
        total_rounds += 1
        update_client_training_parameters(clients, 'epochs', epochs, MAX_EPOCHS, MIN_EPOCHS)
        update_client_training_parameters(clients, 'steps_per_epoch', steps, MAX_STEPS, MIN_STEPS)

        training_time = 0
        client_encrypted_weights = []

        # Select clients using EMD
        num_selected = int(len(clients) * CLIENT_FRACTION)  # Select a fraction of clients
        selected_clients = select_clients_using_emd(clients, num_selected, num_classes)

        # Train each client and collect encrypted weights
        for client in selected_clients:
            print(f"Training client in folder: {client['folder']}")
            client['model'] = clone_model(server['model'])
            client['model'].set_weights(server['model'].get_weights())
            compileModel(client['model'], optimizer, 'binary_crossentropy')

            if client['update']:
                client['model'], client['loss_train'], client['loss_val'], client['round_time'] = trainClientModel(
                    client['model'], client['epochs'], client['training'][0], client['training'][1],
                    client['validation'][0], client['validation'][1], steps_per_epoch=client['steps_per_epoch'])
                client['rounds'] += 1
                training_time = max(training_time, client['round_time'])

                # Encrypt client model weights
                encrypted_weights = encrypt_weights(server_context, client['model'].get_weights())
                client_encrypted_weights.append(encrypted_weights)


        # Print model weights before aggregation
        print("Model Weights Before Aggregation:")
        for i, w in enumerate(server['model'].get_weights()):
            print(f"Layer {i} Shape: {w.shape}, Mean: {np.mean(w):.4f}, Std: {np.std(w):.4f}")

        # Aggregate encrypted weights and decrypt them
        aggregated_encrypted_weights = aggregate_encrypted_weights(client_encrypted_weights)
        original_shapes = [w.shape for w in server['model'].get_weights()]  # Get original shapes
        decrypted_weights = decrypt_weights(server_context, aggregated_encrypted_weights, original_shapes)

        # Print model weights after aggregation
        print("Model Weights After Aggregation:")
        # 🔥 Normalize weights before updating the model
        for i, w in enumerate(decrypted_weights):
            if np.isnan(w).any() or np.isinf(w).any():
                print(f"Warning: NaN detected in layer {i}. Resetting layer to zeros.")
                decrypted_weights[i] = np.zeros(original_shapes[i], dtype=np.float32)
            else:
                decrypted_weights[i] = decrypted_weights[i] / np.max(np.abs(decrypted_weights[i]))  # Normalize values

        # Update model weights
        server['model'].set_weights(decrypted_weights)

        print(f"\n################ Round: {total_rounds:05d} ################")
        f1_val, f1_std_val = select_clients(server['model'], clients, training_mode=training_mode)
        print("==============================================")
        print(f"Average F1 Score: {f1_val}")
        print(f"Std_dev F1 Score: {f1_std_val}")

        # Store validation loss for this round
        avg_val_loss = np.mean([client['loss_val'] for client in clients])  # Average validation loss across clients
        val_losses.append(avg_val_loss)  # Append to the list

        row = {'Model': model_name if f1_val < max_f1 else f"*{model_name}", 'Round': int(total_rounds),
               'AvgF1': f"{f1_val:.5f}"}
        for client in clients:
            row[client['name'] + '(f1)'] = f"{client['f1_val']:.5f}"
            row[client['name'] + '(loss)'] = f"{client['loss_val']:.5f}"
        writer.writerow(row)
        training_file.flush()

        if f1_val > max_f1:
            max_f1 = f1_val
            best_model = clone_model(server['model'])
            best_model.set_weights(server['model'].get_weights())
            print(f"New Max F1 Score: {max_f1}")
            stop_counter = 0
        else:
            stop_counter += 1
            print(f"Stop counter: {stop_counter}")

        print(f"Current Max F1 Score: {max_f1}")
        print("##############################################\n")

        total_client_rounds = sum(client['rounds'] for client in clients)

        f1_val, f1_std_val = assess_best_model(best_model, clients, update_clients=True, print_f1=False)
        row = {'Model': model_name, 'Epochs': epochs, 'Steps': steps, 'Mode': training_mode, 'Weighted': weighted,
               'Experiment': 0, 'ClientsOrder': ' '.join(map(str, client_indices)), 'Round': int(total_rounds),
               'TotalClientRounds': int(total_client_rounds), 'F1': f"{f1_val:.5f}",
               'F1_std': f"{f1_std_val:.5f}", 'Time(sec)': f"{training_time:.2f}"}
        for client in clients:
            row[client['name'] + '(f1)'] = f"{client['f1_val_best']:.5f}"
            row[client['name'] + '(rounds)'] = int(client['rounds'])

        tuning_writer.writerow(row)
        hyperparamters_tuning_file.flush()

        stop = stop_counter > PATIENCE

    best_model_file_name = f"{time_window}t-{max_flow_len}n-{model_name}-global-model.h5"
    best_model.save(outdir + '/' + best_model_file_name)
    training_file.close()
    shutil.move(outdir + '/' + training_filename + '.csv',
                outdir + '/' + training_filename + f"-rounds-{total_rounds}.csv")
    hyperparamters_tuning_file.close()
# We evaluate the aggregated model on the clients validation sets
# in a real scenario, the server would send back the aggregated model to the clients, which evaluate it on their local validation data
# as a final step, the clients would send the resulting f1 score to the server for analysis (such as the weighted avaerage below)
def select_clients(server_model, clients, training_mode):
    average_f1, std_dev_f1 = assess_server_model(server_model, clients, update_clients=True, print_f1=True)

    # selection of clients to train in the next round for fedavg and flddos
    random_clients_list = random.sample(clients, int(len(clients) * CLIENT_FRACTION))

    for client in clients:
        if training_mode == "flad" and client['f1_val'] <= average_f1:
            client['update'] = True
        elif training_mode != "flad" and client in random_clients_list:
            client['update'] = True
        else:
            client['update'] = False

    return average_f1, std_dev_f1


# check the global model on the clients' validation sets
def assess_server_model(server_model, clients, update_clients=False, print_f1=False):
    f1_val_list = []
    for client in clients:
        X_val, Y_val = client['validation']
        Y_pred = np.squeeze(server_model.predict(X_val, batch_size=2048) > 0.5)
        client_f1 = f1_score(Y_val, Y_pred)
        f1_val_list.append(client_f1)
        if update_clients == True:
            client['f1_val'] = f1_score(Y_val, Y_pred)
        if print_f1 == True:
            print(client['name'] + ": " + str(client['f1_val']))

    K.clear_session()
    gc.collect()

    if len(clients) > 0:
        average_f1 = np.average(f1_val_list)
        std_dev_f1 = np.std(f1_val_list)
    else:
        average_f1 = 0
        std_dev_f1 = 0

    return average_f1, std_dev_f1


# check the BEST global model on the clients' validation sets
def assess_best_model(server_model, clients, update_clients=False, print_f1=False):
    f1_val_list = []
    for client in clients:
        X_val, Y_val = client['validation']
        Y_pred = np.squeeze(server_model.predict(X_val, batch_size=2048) > 0.5)
        client_f1 = f1_score(Y_val, Y_pred)
        f1_val_list.append(client_f1)
        if update_clients == True:
            client['f1_val_best'] = f1_score(Y_val, Y_pred)
        if print_f1 == True:
            print(client['name'] + ": " + str(client['f1_val_best']))

    K.clear_session()
    gc.collect()

    if len(clients) > 0:
        average_f1 = np.average(f1_val_list)
        std_dev_f1 = np.std(f1_val_list)
    else:
        average_f1 = 0
        std_dev_f1 = 0

    return average_f1, std_dev_f1


# We dynamically assign the number of training steps/epochs (called 'parameter') to each client for the next training round
# The result is based on the f1 score on the local validation set obtained by each client and communicated to
# the server along with the model update
def update_client_training_parameters(clients, parameter, value, max_value, min_value):
    f1_list = []
    update_clients = []

    # here we select the clients that must be updated
    for client in clients:
        if client['update'] == True:
            update_clients.append(client)

    if value == 'auto':  # dynamic parameter based on f1_val
        # the resulting parameters depend on the current f1 values of the clients that will
        # be updated. Such a list of clients is determined in method average_f1_val
        for client in update_clients:
            f1_list.append(client['f1_val'])

        if len(set(f1_list)) > 1:
            min_f1_value = min(f1_list)
            max_value = max(min_value + 1,
                            math.ceil(max_value * (1 - min_f1_value)))  # min acceptable value for is min_value+1
            value_list = max_value + min_value - scale_linear_bycolumn(f1_list, np.min(f1_list), np.max(f1_list),
                                                                       high=float(max_value), low=min_value)

        elif len(set(f1_list)) == 1:  # if there a single f1 value,  scale_linear_bycolumn does not work
            value_list = [max_value] * len(update_clients)
        else:
            return 0

        for client in update_clients:
            client[parameter] = int(value_list[update_clients.index(client)])
            # print ("Client: " + client['name'] + " F1: " + str(client['f1_val']) + " Parameter(" + parameter + "): "  + str(client[parameter]))


    else:  # static parameter
        for client in update_clients:
            client[parameter] = value

    return len(update_clients)


# FedAvg, with the additional option for averaging without weighting with the number of local samples
def aggregation_weighted_sum(server, clients, weighted=True):
    total = 0

    aggregated_model = clone_model(server['model'])
    aggregated_weights = aggregated_model.get_weights()
    aggregated_weights_list = []

    for weights in aggregated_weights:
        aggregated_weights_list.append(np.zeros(weights.shape))

    weights_list_size = len(aggregated_weights_list)

    for client in clients:
        if weighted == True:
            avg_weight = client['samples']
        else:
            avg_weight = 1
        total += avg_weight
        client_model = client['model']
        client_weights = client_model.get_weights()
        for weight_index in range(weights_list_size):
            aggregated_weights_list[weight_index] += client_weights[weight_index] * avg_weight

    aggregated_weights_list[:] = [(aggregated_weights_list[i] / total) for i in
                                  range(len(aggregated_weights_list))]
    aggregated_model.set_weights(aggregated_weights_list)

    return aggregated_model
