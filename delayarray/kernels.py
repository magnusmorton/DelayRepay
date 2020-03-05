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

gemm_new = """


    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);
    uint widthA=num_cols_A;
    uint widthB=num_cols_B;
    /* Vectorization of input Matrices reduces their width by a factor of 4 */
    widthB /= 4;

    for(int i = 0; i < widthA; i=i+4)
    {{
        float4 tempA0 = {matrixA}[i/4 + (pos.y << TILEY_SHIFT) * (widthA / 
4)];
        float4 tempA1 = {matrixA}[i/4 + ((pos.y << TILEY_SHIFT) + 1) * 
(widthA / 4)];
        float4 tempA2 = {matrixA}[i/4 + ((pos.y << TILEY_SHIFT) + 2) * 
(widthA / 4)];
        float4 tempA3 = {matrixA}[i/4 + ((pos.y << TILEY_SHIFT) + 3) * 
(widthA / 4)];

        //Matrix B is not transposed 
        float4 tempB0 = {matrixB}[pos.x + i * widthB];	
        float4 tempB1 = {matrixB}[pos.x + (i + 1) * widthB];
        float4 tempB2 = {matrixB}[pos.x + (i + 2) * widthB];
        float4 tempB3 = {matrixB}[pos.x + (i + 3) * widthB];

        sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * 
tempB2.x + tempA0.w * tempB3.x;
        sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * 
tempB2.y + tempA0.w * tempB3.y;
        sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * 
tempB2.z + tempA0.w * tempB3.z;
        sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * 
tempB2.w + tempA0.w * tempB3.w;

        sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * 
tempB2.x + tempA1.w * tempB3.x;
        sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * 
tempB2.y + tempA1.w * tempB3.y;
        sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * 
tempB2.z + tempA1.w * tempB3.z;
        sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * 
tempB2.w + tempA1.w * tempB3.w;

        sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * 
tempB2.x + tempA2.w * tempB3.x;
        sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * 
tempB2.y + tempA2.w * tempB3.y;
        sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * 
tempB2.z + tempA2.w * tempB3.z;
        sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * 
tempB2.w + tempA2.w * tempB3.w;

        sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * 
tempB2.x + tempA3.w * tempB3.x;
        sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * 
tempB2.y + tempA3.w * tempB3.y;
        sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * 
tempB2.z + tempA3.w * tempB3.z;
        sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * 
tempB2.w + tempA3.w * tempB3.w;
    }}
    {matrixC}[pos.x + ((pos.y <<  TILEY_SHIFT) + 0) * widthB] = sum0;
    {matrixC}[pos.x + ((pos.y <<  TILEY_SHIFT) + 1) * widthB] = sum1;
    {matrixC}[pos.x + ((pos.y <<  TILEY_SHIFT) + 2) * widthB] = sum2;
    {matrixC}[pos.x + ((pos.y <<  TILEY_SHIFT) + 3) * widthB] = sum3;
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
