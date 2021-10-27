# Copyright (C) 2020 by University of Edinburgh


import delayrepay.backend
from .delayarray import DelayArray

np = delayrepay.backend.backend

def fft(self, *args, **kwargs):
    nargs = [arg.__array__() if isinstance(arg, DelayArray) else arg
             for arg in args]
    return np.fft.fft(*nargs, **kwargs)
