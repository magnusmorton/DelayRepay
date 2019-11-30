'''Delay array and related stuff'''

from dataclasses import dataclass
from typing import List, Any
import numpy as np
import numpy.lib.mixins

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


def vector_add():
    fun = UserFun(String("add"), Array([String("x"), String("y")], String("{ return x+y; }"), Seq([Float(), Float()]), Float()))
    size = SizeVar(String(name="N"))

    return LiftFunction(
        [ArrayTypeWSWC(Float(), size), ArrayTypeWSWC(Float(), size)],
        Lambda([Var("left"), Var("right")],
               Join().compose(MapWrg(
                   Join().compose(MapLcl(
                       MapSeq(fun))).compose(Split(Number(4)))
                   )).compose(Split(Number(1024))).apply(Zip(Var("left"), Var("right")))))
        

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
            pass

class LiftWalker:
    def walk(self, arr):
        pass

    def emit_mm(self, arr):
        return

@dataclass
class LiftNode:
    '''A lift IR node '''

    def compose(self, other: LiftNode) -> Compose:
        '''convenience method for composition'''
        return Compose(self, other)

    def apply(self, other: LiftNode) -> Apply:
        '''convenience method for application'''
        return Apply(self, other)

@dataclass
class LiftType(LiftNode):
    '''A lift type'''

@dataclass
class SizeVar(LiftNode):
    ''' An array sizevar'''
    name: String

@dataclass
class Atom(LiftNode):
    '''Atoms'''

@dataclass
class Seq(LiftType):
    '''Sequence type'''
    types: List[LiftType]

@dataclass
class ArrayTypeWSWC(LiftType):
    '''Funciton input array type'''
    dtype: LiftType
    sizevar: SizeVar

@dataclass
class Float(LiftType):
    '''A float'''

@dataclass
class Int(LiftType):
    '''An Int'''

@dataclass
class Var(Atom):
    '''Not functions/classes'''
    val: str

@dataclass
class Number(Atom):
    '''Numbers'''
    val: float

@dataclass
class String(Atom):
    '''Strings'''
    val: str

@dataclass
class Unary(LiftNode):
    '''Unary function'''
    arg: LiftNode

@dataclass
class Binary(LiftNode):
    '''Binary Function'''
    left: LiftNode
    right: LiftNode


@dataclass
class Compose(Binary):
    '''Composition function'''

@dataclass
class Apply(Binary):
    '''Function application ($)'''

@dataclass
class Map(Unary):
    '''Map base class'''

@dataclass
class MapWrg(Map):
    '''Map function over workgroups?'''

@dataclass
class Join(LiftNode):
    '''Join. Not sure what it does. Barrier? '''

@dataclass
class MapLcl(Map):
    '''Map fuction over work items?'''

@dataclass
class MapSeq(Map):
    '''Map sequentially'''

@dataclass
class Split(Unary):
    '''Split data'''

@dataclass
class Zip(Binary):
    '''Zip'''

@dataclass
class Array(LiftNode):
    '''Lift Array type'''
    members: List[String]

@dataclass
class Lambda(LiftNode):
    '''A scala lambda'''
    args: List[Var]
    body: LiftNode

@dataclass
class UserFun(LiftNode):
    '''User Functions for OpenCL'''
    name: String
    args: Array
    body: String
    arg_type: LiftType
    ret_type: LiftType

@dataclass
class LiftFunction:
    '''A lift fun...'''
    types: List[LiftType]
    body: LiftNode


@dataclass
class NumpyEx:
    '''Numpy expression'''

    @dataclass
class BinaryNumpyEx(NumpyEx):
    '''Binary numpy expression'''
    left: NumpyEx
    right: NumpyEx

@dataclass
class Dot(BinaryNumpyEx):
    '''np.dot func'''
    name: np.ufunc = np.dot

@dataclass
class Multiply(BinaryNumpyEx):
    '''np.multiply'''
    name = np.mulitply

@dataclass
class Add(BinaryNumpyEx):
    '''np.add'''
    name  = np.add

@dataclass
class NPArray(NumpyEx):
    '''ndarray'''
    array: np.ndarray
    

if __name__ == "__main__":
    walker = NumpyWalker()
    arr = array([1,2,3])
    res = (arr @ arr) + arr
    print(res)
    print(walker.walk(res))
