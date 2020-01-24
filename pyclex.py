import pyopencl as cl
import numpy as np


a_np = np.ones(5000000).astype(np.float32)
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)

prg = cl.Program(ctx, """
__kernel void inc(
    __global const float *a_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + 1;
}
""").build()


res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.inc(queue, a_np.shape, None, a_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)


print(res_np)
