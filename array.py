'''Delay array and related stuff'''

from dataclasses import dataclass
from typing import List, Any
import numpy as np
import numpy.lib.mixins
from num import *

def cast(func):
    '''cast to Delay array decorator'''
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
    def __new__(cls, shape, dtype='float64', buffer=None, offset=0,
                strides=None, order=None, parent=None, ops=None, ex=None):
        self = super(DelayArray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
            self.ex = NPArray(buffer)
        elif ops is None:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order)
            self.ex = NPArray(np.ndarray(shape, dtype, buffer, offset, strides, order))
        elif ex is not None:
            self.ex = ex
        else:
            # do type inference
            pass
       
        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        self.ops = ops
        return self

    def __repr__(self):
        return str(self.__array__())


    def child(self, ops):
        return DelayArray(self.shape,  ops=ops)
    
    def walk(self):
        walker = NumpyWalker()
        return walker.walk(self)

    def __array__(self):
        return NumpyFunction(self.ex)()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(ufunc)
        print(method)
        print(inputs)
        if ufunc.__name__ == 'multiply':
            print("FOO")
        cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return DelayArray(self.shape, ops=(ufunc, inputs, kwargs), ex=cls(args[0], args[1]))



    def __array_function__(self, func, types, args, kwargs):
        print("BAR")
        return self



array = cast(np.array)


def arg_to_numpy_ex(arg:Any) -> NumpyEx:
    from numbers import Number
    if isinstance(arg, DelayArray):
        return arg.ex
    elif isinstance(arg, Number):
        return Scalar(arg)
    else:
        print(arg)
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
        

class NumpyWalker:
    '''Simply evaluates the numpy ops'''
    def walk(self, arr):
        '''walk the array'''
        if arr.ops is None:
            print("NONE")
            print(arr._ndarray)
            return arr._ndarray
        return arr.ops[0](self.walk(arr.ops[1][0]), self.walk(arr.ops[1][1]))

class StringWalker:
    def walk(self, arr):
        if arr.ops is None:
            return str(arr._ndarray)
        else:
            strs = ["np." + arr.ops[0].__name__, "("]
            for arg in arr.ops[1]:
                strs.append(self.walk(arg))
            strs.append(")")
            return ''.join(strs)

# class LiftWalker:
#     def walk(self, _):
#         pass

#     def emit_mm(self, _):
#         return


def main():
    ''' main method'''
    walker: Any = NumpyWalker()
    arr = array([1, 2, 3])
    arr2 = array([3,4,5])
    res = (arr @ arr) + arr2
    print(res)
    print(res.ex)
    print(walker.walk(res))
    walker = StringWalker()
    print(walker.walk(res))
    visitor = NumpyVarVisitor()
    print(visitor.visit(res.ex))
    func = NumpyFunction(res.ex)
    print(func())

if __name__ == "__main__":
    main()
