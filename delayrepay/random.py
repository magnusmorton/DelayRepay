# Copyright (C) 2020 by University of Edinburgh

import delayrepay.backend as be
#from numpy.random import *
from .delayarray import cast
np = be.backend.np

rand = cast(np.random.rand)
randn = cast(np.random.randn)
random = cast(np.random.random)
seed = np.random.seed
randint = cast(np.random.randint)
choice = cast(np.random.choice)
