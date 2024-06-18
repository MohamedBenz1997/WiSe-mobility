"""
This is the runner script of the WiSe simulator including mobility management

@authors: Mohamed Benzaghta
"""
# General import
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat
dtype = torch.double

# My simulator import
import tensorflow as tf
from TerrestrialClass import Terrestrial
from DeploymentClass import  Deployment
from SinrClass import SINR
from config import Config
from plot_class import Plot
SINR = SINR()
config = Config()
plot = Plot()

# Specifying antenna parameters: tilts, vHPBW, Tx power
##############################################################################
BS_tilt = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), -12.0, -12.0, tf.float32), axis=0), axis=2)
BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0), axis=2)
BS_HPBW_v = tf.tile(HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0), axis=2)
P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

# Specifying mobility parameters
##############################################################################
T = 15 #Total number of time iterations
UE_speed_base = tf.constant(-50, shape=[2 * config.batch_num, config.Nuser_drop, 1], dtype=tf.float32)
UE_speed_zeros = tf.zeros([2 * config.batch_num, config.Nuser_drop, 1], dtype=tf.float32)
# UE_speed = tf.concat([UE_speed_base, UE_speed_zeros], axis=2)
UE_speed = tf.concat([UE_speed_zeros, UE_speed_base, UE_speed_zeros], axis=2)
CIO_t = 0.0
CIO_s = 0.0
Hys = 0.0
TTT = 0
T310 = 0
Q_out = -34.0
# Define the threshold Ts
Ts = 4000
cell_assoc = "A3"
# Initialization for A3 logic and their place holders
number_of_users = 853 #2 users are UAVs, if I do 100% GUEs simulator crashes
# Counter for each user for A3 condition, initialized to zeros
A3_counter = tf.zeros([number_of_users], dtype=tf.int64)
A3_event_history = tf.TensorArray(dtype=tf.int64, size=T)
# Placeholder for serving BS history, initialized to zeros with a shape determined by T and UEs
best_base_station_indices = tf.zeros([number_of_users], dtype=tf.int64)
serving_BS_history = tf.TensorArray(dtype=tf.int64, size=T)
# Placeholder for HO failures
HO_failure_history = tf.TensorArray(dtype=tf.int64, size=T)
# Initialize T310_event to keep track of when the condition is first triggered
T310_event = tf.zeros([number_of_users], dtype=tf.int32)
# Placeholder to store the time each user spends at each base station for each time step
TimeOfStay = tf.TensorArray(dtype=tf.int32, size=T)
current_stay_time = tf.zeros([number_of_users], dtype=tf.int32)
previous_best_base_station_indices = tf.zeros([number_of_users], dtype=tf.int64)
# Run simulator based on specified parameters
##############################################################################
data = Terrestrial()
data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
data.BS_HPBW_v = BS_HPBW_v
data.UE_speed = UE_speed

for t in range(T):

    data.T = t

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

    # Cell assosiation
    if t == 0 or cell_assoc != "A3":
        # Max RSRP association
        RSRP_served = tf.reduce_max(RSRP, axis=0)
        best_base_station_indices = tf.argmax(RSRP, axis=0)
        RSRP_served_ref = RSRP_served
    else:
        # Implement A3 association logic
        for i in tf.range(RSRP.shape[1]):
            ue_rsrp = RSRP[:, i]
            ue_best_base_station_indices = best_base_station_indices[i]
            candidate_rsrp = tf.reduce_max(ue_rsrp)
            candidate_index = tf.argmax(ue_rsrp)

            condition_1 = (candidate_rsrp + CIO_t - Hys) > (tf.gather(ue_rsrp, ue_best_base_station_indices) + CIO_s)
            condition_2 = candidate_index != ue_best_base_station_indices
            condition = tf.logical_and(condition_1, condition_2)

            # Increment A3_counter if condition is satisfied
            updated_A3_counter = tf.where(condition, A3_counter[i] + 1, A3_counter[i])

            # Store the index value of the base station that triggered the A3 event
            A3_triggered_index = tf.where(condition, candidate_index, ue_best_base_station_indices)

            def update_values():
                updated_A3_counter = tf.constant(0)  # Reset counter after association change
                return ue_rsrp[A3_triggered_index], A3_triggered_index, updated_A3_counter

            def maintain_values():
                return RSRP[tf.cast(best_base_station_indices[i], tf.int32), i], best_base_station_indices[i], updated_A3_counter

            RSRP_served_i, best_base_station_indices_i, A3_counter_i = tf.cond(
                condition & (A3_counter[i] >= TTT), update_values, maintain_values)

            RSRP_served = tf.tensor_scatter_nd_update(RSRP_served, [[i]], [RSRP_served_i])
            best_base_station_indices = tf.tensor_scatter_nd_update(best_base_station_indices, [[i]],
                                                                    [best_base_station_indices_i])
            A3_counter = tf.tensor_scatter_nd_update(A3_counter, [[i]], [A3_counter_i])


    # Track time spent at each base station
    time_increment = tf.ones([number_of_users], dtype=tf.int32)
    current_stay_time = tf.where(tf.equal(best_base_station_indices, previous_best_base_station_indices),
                                 current_stay_time + time_increment, tf.zeros_like(current_stay_time))

    # Store the time of stay for this iteration
    TimeOfStay = TimeOfStay.write(t, current_stay_time)

    # Update previous best base station indices
    previous_best_base_station_indices = best_base_station_indices

    # A3 event parameter update
    A3_event = tf.cast(tf.not_equal(A3_counter, 0), tf.int64)
    A3_event_history = A3_event_history.write(t, A3_event)
    A3_event_history_tensor = A3_event_history.stack()
    A3_event_history_tensor = tf.cast(A3_event_history_tensor, tf.float32)

    # Associated BS parameter update
    serving_BS_history = serving_BS_history.write(t, best_base_station_indices)
    serving_BS_history_tensor = serving_BS_history.stack()
    serving_BS_history_tensor = tf.cast(serving_BS_history_tensor, tf.float32)

    #Time of trigger of A3 event parameter update
    # Identify where an A3 event starts
    A3_event_starts = tf.logical_and(
        tf.concat([tf.zeros([1, number_of_users], dtype=A3_event_history_tensor.dtype),
                   A3_event_history_tensor[1:, :] - A3_event_history_tensor[:-1, :]], axis=0) == 1,
        A3_event_history_tensor == 1
    )
    # Create a multiplier tensor for iteration numbers
    iteration_multiplier = tf.range(T) * 1000 #in ms assuminng my iteration is 1s
    iteration_multiplier = tf.reshape(iteration_multiplier, [-1, 1])
    iteration_multiplier = tf.broadcast_to(iteration_multiplier, tf.shape(A3_event_history_tensor))
    # Apply the multiplier where a new A3 event starts
    TimeOfTrigger_history_tensor = tf.where(A3_event_starts, iteration_multiplier, 0)

    # Time of assosiation parameter update
    # Differences between adjacent elements (BS indexes compared to previous iteration) to detect changes
    changes = tf.concat(
        [tf.zeros([1, number_of_users], dtype=tf.bool),
         tf.not_equal(serving_BS_history_tensor[1:], serving_BS_history_tensor[:-1])],
        axis=0)
    # Generate iteration numbers
    iteration_numbers = tf.range(T, dtype=tf.int32) * 1000
    # Broadcast iteration numbers across all users
    broadcasted_iterations = tf.tile(tf.reshape(iteration_numbers, [-1, 1]), [1, number_of_users])
    # Use changes mask to select iteration numbers where changes occur, otherwise zero
    TimeOfAssociation_history_tensor = tf.where(changes, broadcasted_iterations, tf.zeros_like(broadcasted_iterations))
    TimeOfAssociation_history_tensor = tf.cast(TimeOfAssociation_history_tensor, dtype=tf.int32)

    # HO counter parameter update
    HO_history_tensor = tf.cast(TimeOfAssociation_history_tensor != 0, dtype=tf.int32)

    # HO failure tracking
    condition_triggered = tf.less(RSRP_served, Q_out)

    # Update T310_event if condition is triggered
    T310_event = tf.where(condition_triggered, T310_event + 1, tf.zeros_like(T310_event))

    # Determine HO_failure based on T310_event exceeding T310
    HO_failure = tf.greater_equal(T310_event, T310)

    # Reset T310_event for users where HO_failure is true
    T310_event = tf.where(HO_failure, tf.zeros_like(T310_event), T310_event)

    # Reset A3 counter and re-associate if HO failure
    def handle_ho_failure():
        new_rsrp_served = tf.reduce_max(RSRP, axis=0)
        new_best_base_station_indices = tf.argmax(RSRP, axis=0)
        return new_rsrp_served, new_best_base_station_indices, tf.zeros_like(A3_counter)


    def handle_no_ho_failure():
        return RSRP_served, best_base_station_indices, A3_counter


    RSRP_served, best_base_station_indices, A3_counter = tf.cond(
        tf.reduce_any(HO_failure), handle_ho_failure, handle_no_ho_failure
    )

    # Compare new best_base_station_indices with the previous iteration's base station indices
    previous_best_base_station_indices = serving_BS_history.read(t)

    HO_failure_tensor = tf.cast(tf.not_equal(best_base_station_indices, previous_best_base_station_indices), dtype=tf.int64)

    # Store HO failure history
    HO_failure_history = HO_failure_history.write(t, HO_failure_tensor)
    HO_failure_history_tensor = HO_failure_history.stack()
    HO_failure_history_tensor = tf.cast(HO_failure_history_tensor, tf.float32)

    # After HO failure handling
    if tf.reduce_any(HO_failure):
        # Reset current_stay_time for users with HO failure
        current_stay_time = tf.where(HO_failure, tf.zeros_like(current_stay_time), current_stay_time)

    # Associated BS parameter update
    serving_BS_history = serving_BS_history.write(t, best_base_station_indices)
    serving_BS_history_tensor = serving_BS_history.stack()
    serving_BS_history_tensor = tf.cast(serving_BS_history_tensor, tf.float32)

    # Time of Stay
    TimeOfStay_tensor_binary = TimeOfStay.stack()
    TimeOfStay_tensor_binary = tf.cast(TimeOfStay_tensor_binary, tf.int32)

    def modify_TimeOfStay_tensor(tensor):
        # Transpose the tensor to iterate over columns
        tensor_transposed = tf.transpose(tensor)

        # Initialize the modified tensor with zeros
        modified_tensor_transposed = tf.zeros_like(tensor_transposed, dtype=tf.int32)

        for i in range(tensor_transposed.shape[0]):
            current_row = tensor_transposed[i]
            new_row = tf.zeros_like(current_row)

            zero_indices = tf.where(tf.equal(current_row, 0))
            zero_indices = tf.concat([zero_indices, tf.constant([[current_row.shape[0]]], dtype=tf.int64)], axis=0)

            start = 0
            for j in range(1, len(zero_indices)):
                end = zero_indices[j][0]
                sum_consecutive_ones = tf.reduce_sum(current_row[start:end])
                if start > 0:
                    new_row = tf.tensor_scatter_nd_update(new_row, [[start - 1]], [sum_consecutive_ones])
                start = end + 1

            modified_tensor_transposed = tf.tensor_scatter_nd_update(modified_tensor_transposed, [[i]], [new_row])

        # Transpose back to the original shape
        modified_tensor = tf.transpose(modified_tensor_transposed)

        return modified_tensor*1000

    TimeOfStay_tensor = modify_TimeOfStay_tensor(TimeOfStay_tensor_binary)

    # Ping-pong calculations
    # Create a tensor to track HO_ping_pong events
    HO_ping_pong = tf.zeros_like(TimeOfStay_tensor, dtype=tf.int32)
    # Iterate over each user
    for user in range(serving_BS_history_tensor.shape[1]):
        # Get the serving BS history and time of stay for the current user
        user_serving_BS_history = serving_BS_history_tensor[:, user]
        user_TimeOfStay = TimeOfStay_tensor[:, user]

        # Iterate over each time step except the first and the last one
        for t in range(1, serving_BS_history_tensor.shape[0] - 1):
            # Check if the time of stay is less than the threshold and if the next serving BS index is the same as the previous one
            if user_TimeOfStay[t] < Ts and user_serving_BS_history[t + 1] == user_serving_BS_history[t - 1]:
                # Mark this event as HO_ping_pong
                HO_ping_pong = tf.tensor_scatter_nd_update(HO_ping_pong, [[t, user]], [1])

    # Convert the HO_ping_pong tensor to numpy for better visualization
    HO_ping_pong_numpy = HO_ping_pong.numpy()
    x=HO_ping_pong*HO_history_tensor

#Total number of performed HOs
total_HO = tf.reduce_sum(HO_history_tensor)

# Total number of HO failures
total_HO_failure = tf.reduce_sum(HO_failure_history_tensor)

# # Plotting the UE
users_xyz = Xuser[0, :, :]
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











