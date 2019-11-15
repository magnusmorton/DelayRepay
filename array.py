import numpy as np

class DelayArray:
    def __new__(cls, shape, dtype='float64', buffer=None, offset=0,
                strides=None, order=None, ndarray=None):
        self = super(tparray, cls).__new__(cls)
        if ndarray is not None:
            self._ndarray = ndarray
        else:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
        self.shape = shape
        self.dtype = self._ndarray.dtype
