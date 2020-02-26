import numpy as nup
import delayarray as np

SIZE = 1024000
mat = np.full((SIZE,), 7).astype(nup.float32)

print((mat * 3 + 9).dot(mat))
