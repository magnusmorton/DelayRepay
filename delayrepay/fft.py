""" numpy.fft wrapper

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


import delayrepay.backend
from .delayarray import DelayArray

np = delayrepay.backend.backend

def fft(self, *args, **kwargs):
    nargs = [arg.__array__() if isinstance(arg, DelayArray) else arg
             for arg in args]
    return np.fft.fft(*nargs, **kwargs)
