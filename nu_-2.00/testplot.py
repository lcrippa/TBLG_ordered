# Basic functionality
import numpy as np
import scipy.linalg as lin
import math
from itertools import product
from h5 import *
import pickle

# triqs+CTHYB DMFT functionality
from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
import triqs
from triqs.operators.util import *

# DFTtools interface
from triqs_dft_tools.converters.hk import *
from triqs_dft_tools.sumk_dft import *

# Plotting functionality
import matplotlib
import matplotlib.pyplot as plt
from triqs.plot.mpl_interface import oplot

matplotlib.use("TkAgg")

#
import os
import sys


filename = sys.argv[1]



file = HDFArchive(filename, 'r')


data = file['dmft_output']
Niter = data['iterations'] -1
Giw  = data['G_iw-'+str(Niter)]

print(Giw)

oplot(Giw['up_1'][0,0].imag, x_window = (0,100))
oplot(Giw['up_1'][1,1].imag, x_window = (0,100))
plt.show()

