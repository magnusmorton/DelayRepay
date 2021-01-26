""" CUDA numpy array with delayed-evaluation semantics

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
# Copyright (C) 2020 by Univeristy of Edinburgh

from typing import Dict, List, Tuple

Shape = Tuple[int, int]

InputDict = Dict[str, "BaseFragment"]


class BaseFragment:
    def __init__(self):
        self.name = None
        self.stmts = []
        self._expr = None
        self._inputs = {}

    @property
    def inputs(self) -> InputDict:
        return self._inputs

    @property
    def kernel_args(self) -> InputDict:
        return self._inputs


class Fragment(BaseFragment):
    def __init__(self, name: str, stmts: List[str], inputs: InputDict) -> None:
        self.name = name
        self.stmts = stmts
        self._inputs = inputs
        # self.dtype = np.float32

    def ref(self) -> str:
        return self.name

    # def expr(self) -> str:
    #     return self._expr

    def to_input(self):
        return {self.name: self.node.array}

    # def to_kern(self) -> cupy.ElementwiseKernel:
    #     body = ";\n".join(dedup(self.stmts))
    #     inargs = [f"T {arg}" for arg in self.kernel_args]
    #     kern = cupy.ElementwiseKernel(
    #         ",".join(inargs),
    #         "T out",
    #         f"{body};\nout = {self.name}",
    #         f"delay_repay_{self.name}",
    #     )
    #     return kern


class InputFragment(BaseFragment):
    def __init__(self, name, arr) -> None:
        super().__init__()
        self.name = name
        self._inputs = {self.name: arr}

    def ref(self) -> str:
        return f"{self.name}"

    def expr(self) -> str:
        return f"{self.name}"


class ScalarFragment(BaseFragment):
    def __init__(self, val) -> None:
        super().__init__()
        self.val = val.val
        self.dtype = val.dtype

    def ref(self) -> str:
        return str(self.val)

    def expr(self) -> str:
        return str(self.val)


class ReductionKernel(Fragment):
    pass


def combine_inputs(*args: InputDict) -> InputDict:
    ret = {}
    for arg in args:
        ret.update(arg)
    return ret
