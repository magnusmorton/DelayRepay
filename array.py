import numpy as np
import numpy.lib.mixins
import copy
def cast(func):
    print("casting")
    def wrapper(*args, **kwargs):
        print(func)
        print(args)
        arr = func(*args, **kwargs)
        print(arr)
        if not isinstance(arr,DelayArray):
            print("let's delay")
            arr = DelayArray(arr.shape, buffer=arr)
        return arr
    return wrapper

class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __new__(cls, shape, dtype='float64', buffer=None, offset=0,
                strides=None, order=None, parent=None):
        self = super(DelayArray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
        else if parent is None :
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        return self


    def __init__(self, *args, **kwargs):
        self.ops = []
        
    def __repr__(self):
        return str(self.ops)


    def child(self, ops):
        self.ops = ops
        return DelayArray(self.shape, parent)
    
        

    def __array__(self):
        return self._ndarray

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(ufunc)
        print(method)
        self.ops.append((ufunc, inputs, kwargs))
        return self



    def __array_function__(self, func, types, args, kwargs):
        print("BAR")
        return self


array = cast(np.array)

if __name__ == "__main__":
    arr = array([1,2,3])
    print(arr)
    arr * arr
