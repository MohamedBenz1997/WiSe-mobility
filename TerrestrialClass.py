"""
This is the Main Class of the simulator:
    This is the class of the TN/NTN that deploys BSs/LEO and GUEs/UAVs.
    Calculate the distances in 2D and 3D then associate UE to BSs/LEO such that each
    BS will have at least 1 associated user. The call of TN/NTN classes to calculate
    the LSG are performed here as  well.

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""
import tensorflow as tf
from config import Config
from LargeScaleGainClass import Large_Scale_Gain
from LSG_Class_UAVs import Large_Scale_Gain_drone
from DeploymentClass import  Deployment
import numpy as np
from NTN_LSG_Class import NTN_Large_Scale_Gain
import math

class Terrestrial(Config):

    def __init__(self):
        Config.__init__(self)
        self.Deployment = Deployment()
        self.Large_Scale_Gain_drone=Large_Scale_Gain_drone()


    def call(self):

        Xap,Xuser,D,D_2d,BS_wrapped_Cord,Azi_phi_deg,Elv_thetha_deg,D_UAV, D_2d_UAV, BS_wrapped_Cord_UAV,Azi_phi_deg_UAV, Elv_thetha_deg_UAV, Xap_ref, Xuser_ref = self.Deployment.Call(self.T, self.UE_speed, self.Xap_ref, self.Xuser_ref, self.Xuser)
        if self.UAVs:
            LSL_1, LSG_1, G_Antenna_1, p_LOS_1, pl_1, shadowing_LOS_1, shadowing_NLOS_1 = self.Large_Scale_Gain_drone.call(D[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],D_2d[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],Azi_phi_deg[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],Elv_thetha_deg[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1], self.Zuser,self.BS_tilt, self.BS_HPBW_v)  # GUEs
            LSL_2, LSG_2, G_Antenna_2, p_LOS_2, pl_2, shadowing_LOS_2, shadowing_NLOS_2 = self.Large_Scale_Gain_drone.call(D[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],D_2d[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],Azi_phi_deg[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],Elv_thetha_deg[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop)+1:, 2], axis=1),self.BS_tilt, self.BS_HPBW_v)  # UAVs
            LSG = tf.concat([LSG_1, LSG_2], axis=2)

        if self.sectoring_status:
            Xap_1 = Xap[:, 0:19, :]
            D_2d_1 = D_2d[:, 0:19, :]
            D_1 = D[:, 0:19, :]
            Xap_1 = tf.tile(Xap_1, [1, 3, 1])
            D_2d_1 = tf.tile(D_2d_1, [1, 3, 1])
            D_1 = tf.tile(D_1, [1, 3, 1])
            Xap_2 = Xap[:, 19:26, :]
            D_2d_2 = D_2d[:, 19:26, :]
            D_2 = D[:, 19:26, :]
            Xap = tf.concat([Xap_1, Xap_2], axis=1)
            D_2d = tf.concat([D_2d_1, D_2d_2], axis=1)
            D = tf.concat([D_1, D_2], axis=1)


        #Saving of data to be reported in the Run script
        self.LSG_UAVs_Corridors = LSG[:,0:57,int(self.GUE_ratio * self.Nuser_drop):]
        self.LSG_GUEs = LSG[:, 0:57, 0:int(self.GUE_ratio * self.Nuser_drop)]
        self.Xuser_UAVs = Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 0:2]
        self.Xuser_GUEs = Xuser[:, 0:int(self.GUE_ratio * self.Nuser_drop), 0:2]
        self.D_2d=D_2d
        self.D = D
        self.Xap_ref = Xap_ref
        self.Xuser_ref = Xuser_ref
        self.Xap = Xap
        self.Xuser = Xuser
        return