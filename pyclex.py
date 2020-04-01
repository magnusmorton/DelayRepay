import pyopencl as cl
import numpy as np

SIZE = (160000000,)

a_np = np.ones(SIZE).astype(np.float32)
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


__kernel void parsum(__global float* input, __global float* partial_sums, __local float* localSums)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

   // get me stuff in local mem plsthnx
    localSums[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < group_size; offset <<= 1) {
        int mask = (offset << 1) - 1;
        if ((local_id & mask) == 0) {
            localSums[local_id] += localSums[offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        partial_sums[get_group_id(0)] = localSums[0];

    }

}

__kernel void thac(__global float* input, __global float* res)
{
        int idx = get_global_id(0);
        
        float var1 = input[idx] * 4.0f;
        float var2 = var1 + 76.0f;
        res[idx] = var2;
}

__kernel void expr(__global float* input, __global float* res)
{
        int idx = get_global_id(0);
        
       

        res[idx] = input[idx] * 4.0f + 76.0f;
}

__kernel void madex(__global float4* input, __global float4* res)
{
        int idx = get_global_id(0);
        
       

        res[idx] = mad(input[idx], 4.0, 76.0);
}
""").build()


res_np = np.empty(SIZE).astype(np.float32)
res_g = cl.Buffer(ctx, mf.READ_WRITE, res_np.nbytes)
res_g2 = cl.Buffer(ctx, mf.READ_WRITE, res_np.nbytes)

inshape = (SIZE[0] // 4, )
print(a_np.shape)
prg.madex(queue, inshape, None, res_g, res_g2)

cl.enqueue_copy(queue, res_np, res_g2)

print(res_np)

