'''Delay array and related stuff'''

import logging
from dataclasses import dataclass
from typing import List, Any
import numpy as np
import numpy.lib.mixins
from num import *
from cl import run_gpu

def cast(func):
    '''cast to Delay array decorator'''
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr,DelayArray):
            arr = DelayArray(arr.shape, buffer=arr)
        return arr
    return wrapper

def calc_shape(func, shape1, shape2):
    
    if len(shape1) == 1:
        shape1 = (1,) + shape1
    if len(shape2) == 1:
        shape2 = (1,) + shape2
    if func == 'multiply':
        return (shape1[0], shape2[1])
    if func == 'add':
        return shape1
    if func == 'dot':
        if shape1[0] == 1:
            return (1,)
        else:
            return (shape1[0], shape2[1])

def calc_type(func, type1, type2):
    if 'float64' in (type1,type2):
        return 'float64'
    elif 'float32' in (type1, type2):
        return 'float32'
    elif 'int64' in (type1, type2):
        return 'int64'
    else:
        return type1

    

class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __new__(cls, shape, dtype='float32', buffer=None, offset=0,
                strides=None, order=None, parent=None, ops=None, ex=None):
        self = super(DelayArray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
            self.ex = NPArray(buffer)
        elif ops is None:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
            self.ex = NPArray(self._ndarray)
        elif ex is not None:
            self.ex = ex
        else:
            # do type inference
            pass
       
        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        self.ops = ops
        self._logger = logging.getLogger("delayRepay.arr")
        return self

    def __repr__(self):
        return str(self.__array__())


    def child(self, ops):
        return DelayArray(self.shape,  ops=ops)
    
    def walk(self):
        walker = NumpyWalker()
        return walker.walk(self)

    def __array__(self):
        # return NumpyFunction(self.ex)()
        if isinstance(self.ex, NPArray):
            return self.ex.array
        return run_gpu(self.ex)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._logger.debug("func: {}".format(ufunc))
        self._logger.debug(method)
        self._logger.debug(inputs)
        if ufunc.__name__ == 'multiply':
            self._logger.debug("FOO")
        if ufunc.__name__ == 'matmul':
            return self._dot(inputs, kwargs)
        # cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return DelayArray(self.shape, ops=(ufunc, inputs, kwargs), ex=BinaryNumpyEx(args[0], args[1], ufunc))

    def _dot(self, args, kwargs):
        args = [arg_to_numpy_ex(arg) for arg in args]
        res = np.array(DelayArray(self.shape, ops=(np.dot, args, kwargs), ex=DotEx(args[0], args[1])))
        return np.sum(res)


    def __array_function__(self, func, types, args, kwargs):
        self._logger.debug("array_function")
        self._logger.debug("func: {}".format(func))

        self._logger.debug("args: {}".format(type(args)))
        args = [arg_to_numpy_ex(arg) for arg in args]
        res = np.array(DelayArray(self.shape, ops=(func, args, kwargs), ex=DotEx( args[0], args[1])))
        return np.sum(res)

    def astype(self, *args, **kwargs):
        self._ndarray = self._ndarray.astype(*args, **kwargs)
        if isinstance(self.ex, NPArray):
            self.ex = NPArray(self._ndarray)
        else:
            raise Exception("Dont call astype here")
        return self

    def dot(self, other, out=None):
        return np.dot(self, other)


array = cast(np.array)

ones = cast(np.ones)

full = cast(np.full)


def arg_to_numpy_ex(arg:Any) -> NumpyEx:
    from numbers import Number
    if isinstance(arg, DelayArray):
        return arg.ex
    elif isinstance(arg, Number):
        return Scalar(arg)
    else:
        print(arg)
        print(type(arg))
        raise NotImplementedError

def func_to_numpy_ex(func):
    return {
        'matmul': Matmul,
        'add': Add,
        'multiply': Multiply
        }[func.__name__]


# def vector_add():
#     fun = UserFun(String("add"), Array([String("x"), String("y")], String("{ return x+y; }"), Seq([Float(), Float()]), Float()))
#     size = SizeVar(String(name="N"))

#     return LiftFunction(
#         [ArrayTypeWSWC(Float(), size), ArrayTypeWSWC(Float(), size)],
#         Lambda([Var("left"), Var("right")],
#                Join().compose(MapWrg(
#                    Join().compose(MapLcl(
#                        MapSeq(fun))).compose(Split(Number(4)))
#                    )).compose(Split(Number(1024))).apply(Zip(Var("left"), Var("right")))))
        

# class NumpyWalker:
#     '''Simply evaluates the numpy ops'''
#     def walk(self, arr):
#         '''walk the array'''
#         if arr.ops is None:
#             print("NONE")
#             print(arr._ndarray)
#             return arr._ndarray
#         return arr.ops[0](self.walk(arr.ops[1][0]), self.walk(arr.ops[1][1]))

# class LiftWalker:
#     def walk(self, _):
#         pass

#     def emit_mm(self, _):
#         return


def main():
    ''' main method'''
    arr = array([1, 2, 3])
    arr2 = array([3,4,5])
    #res = (arr @ arr) + arr2
    res = arr + 1
    print(res)
    print(res.ex)
    visitor = NumpyVarVisitor()
    print(visitor.visit(res.ex))
    func = NumpyFunction(res.ex)
    print(func)
    exec(str(func), globals())
    print(jitfunc)
    print(func())

if __name__ == "__main__":
    main()
