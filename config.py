"""
This is the Configuration script of the simulator

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
import math
import numpy as np

class Config():

    def __init__(self):

        # ------------ Define the general paramters that will be used in the upcoming functions.
        self.batch_num = 1   #Number of iterations to run
        self.beta_open_loop = 1
        self.Zuser = 1.5            #Hight of user in m
        self.Zap = 25.0             #Hight of BSs in m
        self.N = 2                  #Number of cell in a tire  N=1 for 7 cells and N=2 for 19 cells
        self.radius = 250           #ISD/2 in meters
        self.fc= 2.0                #TN freq in GHz
        self.fc_Hz = 2.0e9          #TN freq in Hz
        self.c = 3.0e8              #Speed of light m/s
        self.DeploymentType= "Hex"  #Deployment choice, Hex or PPP

        if self.DeploymentType=="Hex":
            self.Nap = 19                          #Number of BSs without sectoring
            if self.N==1:
                self.Nap = 7
            self.Nuser_drop = 15*3*self.Nap        #Number of UEs dropped #use 57 if debugging UE is on

        self.bandwidth = 1e7 #TN Bandwidth in Hz
        self.noise_figure_user= 9 #TN Noise Figure in dB
        self.Dist2D_exclud =0.0 #In mobility I disabled this, work needs to be done if included

        self.sectoring_status = True        #To sectorize each physical BS position to 3 sectors, 120 degree

        # ------------ TN BSs power/noise parameters
        self.Ptx_dBm = 46.0
        self.noise_db = -174+10*tf.math.log(self.bandwidth)/tf.math.log(10.0)   #TN Noise in dBm
        self.noise_figure_user = 9  #UE noise figure in dB

        # ------------ UE power/noise parameters
        self.UE_P_Tx_dbm = 23.0         #TN UE Tx power in dBm
        self.UE_P_Tx_sat_dbm = 23.0     #NTN UE Tx power in dBm
        self.UE_bandwidth= 360.0e3      #UE Bandwidth
        self.UE_noise_db=-174+10*tf.math.log(self.UE_bandwidth)/tf.math.log(10.0)
        self.UE_P_over_noise_db = self.UE_P_Tx_dbm - self.UE_noise_db
        self.UE_P_over_noise_db_sat = self.UE_P_Tx_sat_dbm - self.UE_noise_db

        # ------------ Fractional Power Control (FPC) for TN UEs
        self.FPC = True
        if self.FPC:
            self.Po=-85.0
            self.alpha=0.8

        # ------------ UAVs deployment
        self.UAVs = True
        # self.Zuav=150.0        #UAV Height
        self.GUE_ratio = 0.9988 #Case5: 0.6667, Case4: 0.8, Case3: 0.93334, Case2:0.993334 #Use 1 and 0 in case of UE debugging
        self.UAV_ratio =  0.0012 #Case5: 0.3333, Case4: 0.2, Case3: 0.06666, Case2:0.006666
        self.Zuav = tf.random.uniform([2 * self.batch_num, int(self.UAV_ratio * self.Nuser_drop)+1, 1], 150.0, 150.0) #+1 for case5,3,2
        self.UAVs_highway = True

        # --------- Corridor hights
        self.h_corr1 = 300.0
        self.h_corr2 = 300.0
        self.h_corr3 = 300.0
        self.h_corr4 = 300.0

        return


