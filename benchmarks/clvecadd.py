import importlib
np = importlib.import_module('cupy')
SIZE = 10240
mat = np.full((SIZE,SIZE), 7).astype(np.float32)

print(mat* 3)
