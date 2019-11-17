import numpy as np



def cast(func):
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr,DelayArray):
            arr = DelayArray(arr.shape, buffer=arr)
        return arr
    return wrapper

class DelayArray:
    def __new__(cls, shape, dtype='float64', buffer=None, offset=0,
                strides=None, order=None):
        self = super(tparray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
        else:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
        self.shape = shape
        self.dtype = self._ndarray.dtype


    def __repr__(self):
        return "DELAYED"

    def __array__(self):
        return self._ndarray

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(ufunc)
        print(method)
        return ufunc(*inputs, **kwargs)


array = cast(np.array)
