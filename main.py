# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy.misc import imsave
import cv2


class MovieWriter(object):
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        self.vout.write(frame)

    def close(self):
        self.vout.release() 
        self.vout = None


CELL_W_SIZE = 64
ncells = CELL_W_SIZE * CELL_W_SIZE

dirs = np.array([[0.0, np.pi/2.0],
                 [np.pi, 3*np.pi/2.0]])
dirs = np.tile(dirs, (CELL_W_SIZE//2, CELL_W_SIZE//2))
dirs = np.reshape(dirs.T, (-1))

dir_vects = np.vstack([np.cos(dirs), np.sin(dirs)])

x = np.arange(0, CELL_W_SIZE) - (CELL_W_SIZE-1) / 2.0

X,Y = np.meshgrid(x,x)

x = np.vstack([np.reshape(X.T, (-1)), np.reshape(Y.T, (-1))])
# (2, 16)

# Sets length of field shift in recurrent connections
cell_spacing = Y[1,0] - Y[0,0]

# Offset of center of inhibitory output
ell = 2 * cell_spacing

# Distance from (0,0) for A below
cell_dists = np.linalg.norm(x.T, 2, axis=1)
# (16,)

a = 1                          # if >1, uses local excitatory connections
lambda_ = 13                   # approx the periodicity of the pattern
beta = 3 / (lambda_ * lambda_) # width of the excitatory region
gamma = 1.1 * beta             # how much inhibitory region wider than excitatory 

W = []

for i in range(ncells):
    tmp = np.tile(np.reshape(x[:,i], (2,1)), (1, ncells))
    # (2,16)
    shifts = tmp - x - ell * dir_vects
    # (2,16)

    # TODO: 無駄が多い?
    squared_shift_lengths = np.linalg.norm(shifts.T, 2, axis=1) ** 2
    # (16,)
    
    tmp = a * np.exp(-gamma * squared_shift_lengths) - np.exp(-beta * squared_shift_lengths)
    # (16,)
    W.append(tmp)


W = np.vstack(W)

R  = np.sqrt(ncells)/2   # radius of main network, in cell-position units
a0 = np.sqrt(ncells)/32  # envelope fall-off rate
dr = np.sqrt(ncells)/2   # radius of tapered region, in cell-position units

A = (((cell_dists) - R + dr) / dr)
A = np.exp(-a0 * A * A)
# (16,)

#non_tapered_inds = find(cellDists < (R-dr));
#A(nonTaperedInds) = 1

dt = 0.5 # time step, ms
tau = 10.0
alpha = 50.0

s = np.random.uniform(low=0.0, high=1.0, size=[ncells])
# (16,)

cmap = matplotlib.cm.get_cmap('jet')

images = []

save_interval = 20

writer = MovieWriter("out.mov", (64,64), 15)

v_constant = np.array([0.0005, 0.0005])

stabilization_time = 100 # no-velocity time for pattern to form, ms

for i in range(3000):
    # Feedforward input
    if i < stabilization_time:
        v = np.array([0.0, 0.0])
    else:
        v = v_constant
    
    B = A * (1 + alpha * np.dot(dir_vects.T, v))
    # (16,)
    
    # Total synaptic driving currents
    s_inputs = np.dot(W, s.T).T + B
    
    # Synaptic drive only increases if input cells are over threshold 0
    s_inputs = np.maximum(s_inputs, 0.0)
    
    # Synaptic drive decreases with time constant tau
    s = s + dt * (s_inputs - s) / tau
    
    if i % save_interval == 0:
        image = np.reshape(s, [CELL_W_SIZE, CELL_W_SIZE])
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = cmap(image)
        image = np.uint8(image[:,:,0:3] * 255)
        #index = i // save_interval
        #imsave("grid{0:03}.png".format(index), image)
        writer.add_frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
writer.close()
