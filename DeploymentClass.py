"""
This is the Deployment Class of the simulator:
    In this class, the cells and UEs (GUEs/UAVs, Indoor/Outdoor) are deployed.
    Distances 2D/3D and Azimuth/Elevation angles are computed in here.

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
from config import Config
import math
from pppclass import DeployPPP
from hexcalss import DeployHex


class Deployment(Config):
    def __init__(self):
        Config.__init__(self)
        if self.DeploymentType=="Hex":
            self.Deploy= DeployHex()
        elif self.DeploymentType=="PPP":
            self.Deploy = DeployPPP()


    def Call(self, T, UE_speed, Xap_ref, Xuser_ref, Xuser):

        if T == 0:
            Xap_ref, Xuser_ref = self.Deploy.call()
            # Xuser_ref = Xuser_ref
            # For debug
            Xuser_ref = Xuser_ref*0.0
            Xuser_ref_x = Xuser_ref[:,:,0:1]
            Xuser_ref_y = Xuser_ref[:, :, 1:2]+950.0
            Xuser_ref_z = Xuser_ref[:, :, 2:]+1.5
            Xuser_ref = tf.concat([Xuser_ref_x, Xuser_ref_y, Xuser_ref_z], axis=2)
            Xap = Xap_ref
            Xuser = Xuser_ref
        else:
            Xap = Xap_ref
            Xuser = Xuser +UE_speed

        
        if self.DeploymentType =="Hex":

            D, D_2d, BS_wrapped_Cord, Xuser = self.Deploy.Dist(Xap, Xuser)
            Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord, Xuser, D_2d)
            D_UAV=0.0
            D_2d_UAV=0.0
            BS_wrapped_Cord_UAV=0.0
            Azi_phi_deg_UAV=0.0
            Elv_thetha_deg_UAV=0.0

        return Xap,Xuser,D,D_2d,BS_wrapped_Cord,Azi_phi_deg,Elv_thetha_deg,D_UAV, D_2d_UAV, BS_wrapped_Cord_UAV,Azi_phi_deg_UAV, Elv_thetha_deg_UAV, Xap_ref, Xuser_ref

    def Azi_Elv_Angles(self,BS_wrapped_Cord,Xuser,D_2d):
        
        Xuser=tf.expand_dims(Xuser,axis=1)
        
        x_diff = BS_wrapped_Cord[:,:,:,0]-Xuser[:,:,:,0]
        y_diff = BS_wrapped_Cord[:,:,:,1]-Xuser[:,:,:,1]
        z_diff = BS_wrapped_Cord[:,:,:,2]-Xuser[:,:,:,2]
        
        Azi_phi_deg=tf.math.atan2(y_diff,x_diff)*180/math.pi 
        Elv_thetha_deg=(tf.math.atan2(z_diff,D_2d)*180.0/math.pi)+90.0
        
        return Azi_phi_deg,Elv_thetha_deg