""" numpy.random wrapper

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
# Copyright (C) 2020 by University of Edinburgh

import cupy.random
#from numpy.random import *
from .cuarray import cast

rand = cast(cupy.random.rand)
randn = cast(cupy.random.randn)
random = cast(cupy.random.random)
seed = cupy.random.seed
randint = cast(cupy.random.randint)
choice = cast(cupy.random.choice)
