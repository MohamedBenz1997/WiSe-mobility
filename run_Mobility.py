import torch
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

## Running WiSE simulator
##############################################################################

# Specifiying antenna paramters: tilts, vHPBW, Tx power
BS_tilt = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), -12.0, -12.0, tf.float32), axis=0), axis=2)
BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0), axis=2)
BS_HPBW_v = tf.tile(HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])

Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0), axis=2)
P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

# Specifiying mobility paramters
T = 5
UE_speed = 3.0 # Generate speed in m/s
CIO_t = -4.0
CIO_s = 0.0
TTT = 0.0 #in seconds
Hys = 0.0 #in dB
cell_assoc = "A3"
total_handovers = 0
##############################################################################

# Run simulator based on specified paramters
data = Terrestrial()
deploy = Deployment()
data.alpha_factor = 0.0  # LEO at 90deg
data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
data.BS_HPBW_v = BS_HPBW_v
deploy.T = T
deploy.UE_speed = UE_speed
data.call()

# Import of the UAVs and GUEs LSG and SINR data
LSG_GUEs = data.LSG_GUEs
sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)



