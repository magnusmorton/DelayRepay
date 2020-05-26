import cupy.fft
from .cuarray import DelayArray

def fft(self, *args, **kwargs):

    print(args)
    nargs = [arg.__array__() if isinstance(arg, DelayArray) else arg
            for arg in args]
    print(nargs)
    return cupy.fft.fft(*nargs, **kwargs)
    
