#include <stdio.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cstdlib>
#include <conio.h>

__constant__ float cosine[64];
__constant__ float coef[8];
__constant__ float cosine2[64];


__global__ void kernel2(int *d_a)                              ///Inverse Discrete Cosine Transform
{
	__shared__ float arr[8][8];
	__shared__ float out[8][8];
	__shared__ float in[8][8];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	arr[ty][tx] = d_a[(bx)*64 + ty*8 + tx];

	__syncthreads();

	
	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine2[tx + k*8]*coef[tx];
		out[ty][k] = in[ty][0]+in[ty][1]+in[ty][2]+in[ty][3]+
						in[ty][4]+in[ty][5]+in[ty][6]+in[ty][7];
	}
	
	arr[ty][tx] = out[ty][tx];
	__syncthreads();
	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine2[ty + k*8]*coef[ty];
		out[k][tx] = in[0][tx]+in[1][tx]+in[2][tx]+in[3][tx]+
						in[4][tx]+in[5][tx]+in[6][tx]+in[7][tx];
	}

	
	d_a[(bx)*64 + ty*8 + tx] = (out[ty][tx]);

}

__global__ void kernel(int *d_a)                               ////Discrete Cosine Transform
{
	__shared__ float arr[8][8];
	__shared__ float out[8][8];
	__shared__ float in[8][8];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	arr[ty][tx] = d_a[(by*768 + bx)*64 + ty*8 + tx];

	__syncthreads();
	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine[tx + k*8];
		out[ty][k] = in[ty][0]+in[ty][1]+in[ty][2]+in[ty][3]+
						in[ty][4]+in[ty][5]+in[ty][6]+in[ty][7];
		out[ty][k]*=coef[k]; 
	}
	arr[ty][tx] = out[ty][tx];
	__syncthreads();
	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine[ty + k*8];
		out[k][tx] = in[0][tx]+in[1][tx]+in[2][tx]+in[3][tx]+
						in[4][tx]+in[5][tx]+in[6][tx]+in[7][tx];
		out[k][tx]*=coef[k];
	}
	__syncthreads();
	d_a[(bx)*64 + ty*8 + tx] = out[ty][tx];

}

int main()
{

	int *a,*d_a,*c;
	float cosval[64],cosval2[64];
	float cons[8];

	cons[0] = sqrt(1.0/8);
	for(int i=1;i<8;i++)
		cons[i] = 0.5;

	a = (int*)malloc(64*sizeof(int));
	c = (int*)malloc(64*sizeof(int));


	for(int i=0;i<8;i++)
		for(int j=0;j<8;j++)
			cosval[i*8 + j] = cos(((2*j + 1)/16.0)*3.14*i);
	for(int i=0;i<8;i++)
		for(int j=0;j<8;j++)
			cosval2[i*8 + j] = cos(((2*i + 1)/16.0)*3.14*j);


	////////////////input values///////////////
	
	for(int i=0;i<8;i++)
		for(int j=0;j<8;j++)
			a[i*8 + j] = i+j;
	//////////////////////////////////////////


	cudaMalloc((void**)&d_a,64*sizeof(int));
	cudaMemcpy(d_a,a,64*sizeof(int),cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(cosine,cosval,64*sizeof(float));
	cudaMemcpyToSymbol(coef,cons,8*sizeof(float));
	cudaMemcpyToSymbol(cosine2,cosval2,64*sizeof(float));


	dim3 dimGrid(1,1);
	dim3 dimBlock(8,8);

	kernel<<<dimGrid,dimBlock>>>(d_a);                  //implements DCT

	cudaMemcpy(c,d_a,64*sizeof(float),cudaMemcpyDeviceToHost);
	
	////////////////////print output of DCT/////////////////
	for(int i=0;i<8;i++){
		for(int j=0;j<8;j++){
			printf(" %d ",c[i*8 + j]);}
		printf("\n");}

	printf(" DCT done\n");

	printf("////////////////////\n");
	
	kernel2<<<dimGrid,dimBlock>>>(d_a);                   //implements IDCT
	
	cudaMemcpy(c,d_a,64*sizeof(float),cudaMemcpyDeviceToHost);

	////////////////////print output of IDCT/////////////
	for(int i=0;i<8;i++){
		for(int j=0;j<8;j++){
			printf(" %d ",c[i*8 + j]);}
		printf("\n");}


	cudaFree(d_a);
	free(a);
	getch();
}












