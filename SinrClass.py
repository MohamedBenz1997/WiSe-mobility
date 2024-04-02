"""
This is the SINR Class of the simulator:
    This is the class in which DL and UL for both TN/NTN SINRs are computed

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
from config import Config
from TerrestrialClass import Terrestrial

class SINR(Config):
    def __init__(self):
        Config.__init__(self)
        self.NF_TN_UL_dB=5.0
        self.NF_TN_UL=tf.math.pow(10.0, self.NF_TN_UL_dB/ 10.0)
        self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
        self.BW_TN = self.bandwidth
        self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)

    data = Terrestrial()

    # ------------ DL SINR
    def sinr_TN(self, LSG, P_Tx_TN):
        if self.N==1:
            P_Tx_TN = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 21, tf.shape(LSG)[-1]])
        elif self.N==2:
            P_Tx_TN = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 57, tf.shape(LSG)[-1]])
        P_Tx_TN = tf.math.pow(10.0, (P_Tx_TN - 30.0) / 10.0)  # Conver to linear
        LSL_TN=-LSG
        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSL_TN_sorted = tf.sort(LSL_TN, axis=1)

        #Finding the BSs ids of served UAVs
        indexing = tf.cast(-LSG == LSL_min_TN, "float32")
        if self.N==1:
            BSs_id = tf.expand_dims(tf.expand_dims(tf.range(0, 21, dtype=tf.float32), 0), 2) * indexing
        elif self.N==2:
            BSs_id = tf.expand_dims(tf.expand_dims(tf.range(0, 57, dtype=tf.float32), 0), 2) * indexing
        BSs_id = tf.reduce_sum(BSs_id, axis=1)

        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
        LSG_min_TN=LSG_min_TN*tf.cast(LSG_min_TN != 1.0, "float32")
        LSG_linear = tf.math.pow(10, (LSG) / 10)

        LSG_TN_sorted=-LSL_TN_sorted
        LSG_TN_sorted = tf.math.pow(10, (LSG_TN_sorted) / 10)
        LSG_TN_sorted = LSG_TN_sorted * tf.cast(LSG_TN_sorted != 1.0, "float32")

        ##If TN relief SINR is need uncomment this line,13 cells of for UAVs at 150m, 11 cells of for UAVs at 100m, 7 cells of for UAVs at 50m
        # LSG_TN_sorted = LSG_TN_sorted[:, 11:, :] * tf.cast(LSG_TN_sorted[:, 11:, :] != 1.0, "float32") #For TN interference relief
        snr_link = LSG_linear * P_Tx_TN / (self.BW_TN * self.NF_TN)
        P_Tx_TN_assigned = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN * tf.cast(LSG_linear == tf.tile(LSG_min_TN, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN = LSG_min_TN * P_Tx_TN_assigned / (self.BW_TN * self.NF_TN)
        denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) - num_TN + self.PSD
        # denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) + self.PSD #For TN interference relief
        sinr_TN_withZeros = num_TN / denom_TN
        #To ignore the zeros which corresponds to the GUEs columns when dealing with UAVs and vice versa
        # bool_mask = tf.not_equal(sinr_TN_withZeros, 0)
        # sinr_TN = tf.boolean_mask(sinr_TN_withZeros, bool_mask)

        sinr_TN = sinr_TN_withZeros
        return sinr_TN


