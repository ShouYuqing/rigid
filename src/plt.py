import os
import sys
import glob

# third party
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../ext/neuron')
from neuron import plot

img = np.load("../data/test_seg.npz")
volume = img['vol_data']
print(volume.shape)
plt.figure()
plot(volume[100, :, :])
plt.savefig("1.png")
