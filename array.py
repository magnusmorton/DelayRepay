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
                strides=None, order=None, parent=None, ops=None):
        self = super(DelayArray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
        elif ops is None:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
        else:
            # do type inference
            pass
        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        self.ops = ops
        return self

    def __repr__(self):
        return "delayarr: " + str(self.dtype)


    def child(self, ops):
        return DelayArray(self.shape,  ops=ops)
    
    def walk(self):
        walker = NumpyWalker()
        return walker.walk(self)

    def __array__(self):
        return self._ndarray

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(ufunc)
        print(method)
        print(inputs)
        return self.child((ufunc, inputs, kwargs))



    def __array_function__(self, func, types, args, kwargs):
        print("BAR")
        return self


array = cast(np.array)

class NumpyWalker:
    def walk(self, arr):
        if arr.ops is None:
            print("NONE")
            print(arr._ndarray)
            return arr._ndarray
        else:
            return arr.ops[0](self.walk(arr.ops[1][0]), self.walk(arr.ops[1][0]))

class StringWalker:
    def walk(self, arr):
        if arr.ops is None:
            return str(arr._ndarray)
        else:
            pass


class LiftFunction:
    def __init__(self, body, *types):
        self.types = types

if __name__ == "__main__":
    walker = NumpyWalker()
    arr = array([1,2,3])
    res = (arr @ arr) + arr
    print(res)
    print(walker.walk(res))
