# Copyright (C) 2020 by University of Edinburgh

#from numpy import *
import delayrepay.backend
from .delayarray import *
import delayrepay.random

import delayrepay.fft

if delayrepay.backend.backend.__name__ == 'cupy':
    import cupy
    cuda = cupy.cuda


fft = delayrepay.fft

pi = delayrepay.backend.backend.np.pi
