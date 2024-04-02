import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import matplotlib.pyplot as plt
dtype = torch.double

#My simulator import
import tensorflow as tf
from TerrestrialClass import Terrestrial
from DeploymentClass import  Deployment
from SinrClass import SINR
from config import Config
from plot_class import Plot
SINR = SINR()
config = Config()
plot = Plot()
from scipy.io import savemat

## Running WiSE simulator including mobility
##############################################################################

# Specifiying antenna paramters: tilts, vHPBW, Tx power
BS_tilt = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), -12.0, -12.0, tf.float32), axis=0), axis=2)
BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0), axis=2)
BS_HPBW_v = tf.tile(HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])

Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0), axis=2)
P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

# Specifiying mobility paramters
T = 25
UE_speed_base = tf.constant(10, shape=[2 * config.batch_num, config.Nuser_drop, 1], dtype=tf.float32)
UE_speed_zeros = tf.zeros([2 * config.batch_num, config.Nuser_drop, 2], dtype=tf.float32)
UE_speed = tf.concat([UE_speed_base, UE_speed_zeros], axis=2)
# UE_speed = tf.concat([UE_speed_zeros, UE_speed_base, UE_speed_zeros], axis=2)
CIO_t = -2.0
CIO_s = 0.0
TTT = 0.0 #in seconds
Hys = 0.0 #in dB
cell_assoc = "A3"
total_handovers_per_iteration = []

# Initialization for A3 logic and their place holders
# Counter for each user for A3 condition, initialized to zeros
A3_counter = tf.zeros([P_Tx_TN.shape[2]], dtype=tf.float32)
# Placeholder for serving BS history, initialized to zeros with a shape determined by T and UEs
number_of_users = 853
best_base_station_indices = tf.zeros([number_of_users], dtype=tf.int64)
serving_BS_history = tf.TensorArray(dtype=tf.int64, size=T)
# Placeholder to track previous iteration's best base station indices
# Initialize with invalid indices (-1) to indicate no base station associated at the start
prev_best_base_station_indices = tf.fill([number_of_users], -1)
# Counter for the total handovers per iteration
# Initialize with zeros for all iterations; update during each iteration based on conditions
total_handovers_per_iteration = tf.Variable(tf.zeros([T], dtype=tf.int32))
##############################################################################

# Run simulator based on specified paramters
data = Terrestrial()
data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
data.BS_HPBW_v = BS_HPBW_v
data.UE_speed = UE_speed

for t in range(T):

    data.T = t
    handovers_this_iteration = 0
    # Initial locations
    if t == 0:
        data.Xap_ref = 0.0
        data.Xuser_ref = 0.0
        data.Xuser = 0.0
        data.call()
        Xap_ref = data.Xap_ref
        Xuser_ref = data.Xuser_ref
        LSG_GUEs = data.LSG_GUEs
        RSRP = config.Ptx_dBm + LSG_GUEs[0,:,:]
        Xuser = data.Xuser

    else:
        # Move users according to their speed
        data.Xap_ref = Xap_ref
        data.Xuser_ref = Xuser_ref
        data.Xuser = Xuser
        data.call()
        Xuser = data.Xuser
        LSG_GUEs = data.LSG_GUEs
        RSRP = config.Ptx_dBm + LSG_GUEs[0, :, :]

    ## Cell assosiation
    # RSRP calculation
    # Association method
    if t == 0 or cell_assoc != "A3":
        RSRP_served = tf.reduce_max(RSRP, axis=0)
        best_base_station_indices = tf.argmax(RSRP, axis=0)
        RSRP_served_ref = RSRP_served
    else:
        # Implement A3 association logic
        for i in tf.range(RSRP.shape[1]):  # TensorFlow prefers tf.range for compatibility with graph mode
            ue_rsrp = RSRP[:, i]
            ue_best_base_station_indices = best_base_station_indices[i]
            candidate_rsrp = tf.reduce_max(ue_rsrp)
            candidate_index = tf.argmax(ue_rsrp)

            condition_1 = (candidate_rsrp + CIO_t - Hys) > (tf.gather(ue_rsrp, ue_best_base_station_indices) + CIO_s)
            condition_2 = candidate_index != ue_best_base_station_indices

            # Then, use tf.logical_and to combine these conditions
            condition = tf.logical_and(condition_1, condition_2)


            def update_values():
                updated_A3_counter = tf.constant(0)  # Reset counter after association change
                return candidate_rsrp, candidate_index, handovers_this_iteration, updated_A3_counter


            def maintain_values():
                updated_A3_counter = A3_counter[i] + 1
                return RSRP_served[i], best_base_station_indices[i], handovers_this_iteration, updated_A3_counter


            RSRP_served_i, best_base_station_indices_i, handovers_this_iteration, A3_counter_i = tf.cond(
                condition & (A3_counter[i] >= TTT), update_values, maintain_values)

            # Since TensorFlow tensors are immutable, use tensor_scatter_nd_update for updating
            RSRP_served = tf.tensor_scatter_nd_update(RSRP_served, [[i]], [RSRP_served_i])
            best_base_station_indices = tf.tensor_scatter_nd_update(best_base_station_indices, [[i]],
                                                                    [best_base_station_indices_i])
            A3_counter = tf.tensor_scatter_nd_update(A3_counter, [[i]], [A3_counter_i])

    serving_BS_history = serving_BS_history.write(t, best_base_station_indices)
    serving_BS_history_tensor = serving_BS_history.stack()

    if t > 0:
        # Count how many users have changed their base station index
        changes = tf.not_equal(best_base_station_indices, prev_best_base_station_indices)
        handovers_this_iteration = tf.reduce_sum(tf.cast(changes, tf.int32))

        # Update the total handovers for this iteration
        total_handovers_per_iteration[t].assign(handovers_this_iteration)

    # Write the current best_base_station_indices to serving_BS_history
    serving_BS_history = serving_BS_history.write(t, best_base_station_indices)

    # Update prev_best_base_station_indices to current for the next iteration's comparison
    prev_best_base_station_indices = best_base_station_indices

    # # Plotting the UE
    # users_xyz = Xuser[0, :, :]
    #
    # # Extract x and y coordinates
    # x_coordinates = users_xyz[:, 0]
    # y_coordinates = users_xyz[:, 1]
    #
    # # Convert tensors to numpy arrays for plotting, if running in TensorFlow 2.x and eager execution is enabled
    # x_coordinates = x_coordinates.numpy()
    # y_coordinates = y_coordinates.numpy()
    #
    # # Plotting
    # plt.figure(figsize=(10, 8))  # Optional: Adjust figure size
    # plt.scatter(x_coordinates, y_coordinates, c='blue', marker='.', label='Users')  # Plot as blue dots
    # plt.title('User Coordinates')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.grid(True)  # Optional: Show grid
    # plt.show()

# After the loop, convert the TensorArray to a regular Tensor
serving_BS_history_tensor = serving_BS_history.stack()

# Convert the total_handovers_per_iteration to a tensor (if needed for further processing)
total_handovers_per_iteration_tensor = total_handovers_per_iteration.value()
total_handovers = tf.reduce_sum(total_handovers_per_iteration.value())

x = serving_BS_history













