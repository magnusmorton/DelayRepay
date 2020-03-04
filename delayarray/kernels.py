# =================================================================================================
# This file contains code extracted from the CLBlast project. The project is
# licensed under Apache Version 2.0. This project loosely follows the Google C++
# styleguide and uses a tab-size of two spaces and a max- width of 100
# characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
# Modified by:
#   Magnus Morton <magnus.morton@ed.ac.uk>
#
# =================================================================================================

xasum = """
// The main reduction kernel, performing the loading and the majority of the operation
__kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void Xasum(const int n,
           const __global real* restrict xgm, const int x_offset, const int x_inc,
           __global real* output) {
  __local real lm[WGS1];
  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);
  // Performs loading and the first steps of the reduction
  real acc;
  SetToZero(acc);
  int id = wgid*WGS1 + lid;
  while (id < n) {
    real x = xgm[id*x_inc + x_offset];
    #if defined(ROUTINE_SUM) // non-absolute version
    #else
      AbsoluteValue(x);
    #endif
    Add(acc, acc, x);
    id += WGS1*num_groups;
  }
  lm[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  // Performs reduction in local memory
  for (int s=WGS1/2; s>0; s=s>>1) {
    if (lid < s) {
      Add(lm[lid], lm[lid], lm[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


"""


mag_sum = """
int local_id = get_local_id(0);
int group_size = get_local_size(0);

__local float localSums[64];
localSums[local_id] = {};
barrier(CLK_LOCAL_MEM_FENCE);
for (int offset = 1; offset < group_size; offset <<= 1) {{
    int mask = (offset << 1) - 1;
    if ((local_id & mask) == 0) {{
        localSums[local_id] += localSums[offset];
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
}}
if (local_id == 0) {{
    output[get_group_id(0)] = localSums[0];

}}
"""


"""
//spawn N threads in only dimension 0, N is dividable by num_rows_A
//like if N is 16, then valid number is 1,2,4,8,16
__kernel void naive_matrix_vector_mul(const __global float * restrict A,
			      const __global float * restrict B,
			      __global float * restrict C,
			      const int num_rows_A,
			      const int num_cols_A
			      ) {
"""
gemv = """
	// make sure spawn a good number of threads, so can be cut evenly
	size_t const num_rows_per_thread = num_rows_A / get_global_size(0);
	size_t const row_start = get_global_id(0) * num_rows_per_thread ;
	size_t const row_end = ( get_global_id(0) + 1 ) * num_rows_per_thread ;

	for(int i = row_start; i < row_end; ++i ) {{
		{}[i] = 0; //C
		for( int j = 0; j < num_cols_A; ++j )
			 {}[i] += {}[ i * num_cols_A + j ] * {}[j]; // C A B
	}}	
"""

gemm = """
	size_t work_group_id = get_group_id(0);
	size_t local_thread_id = get_local_id(0);

	float res = 0;
	for( int j = 0; j< num_cols_A ; ++j )
		res +=  {}[ work_group_id * num_cols_A + j ] * {}[ local_thread_id + j * num_rows_A ];

        {}[work_group_id * num_cols_A + local_thread_id] = res;
"""

"""
//each work group handle a row,
//each thread in the work group hand a col
//so spawn num_rows_A blocks, and num_cols_A threads
kernel void naive_matrix_vector_mul(const global float * restrict A,
			      const global float * restrict B,
			      global float * restrict C,
			      const int num_rows_A,
			      const int num_cols_A
			      ) {
	size_t work_group_id = get_group_id(0);
	size_t local_thread_id = get_local_id(0);

	float res = 0;
	for( int i = 0; i< num_cols_A ; ++i )
		res +=  {}[ work_group_id * num_cols_A + i ] * {}[ local_thread_id + i * num_rows_A ];

        {}[work_group_id * num_cols_A + local_thread_id] = res;
"""
