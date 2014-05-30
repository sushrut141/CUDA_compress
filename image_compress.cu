#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv\cv.h>
#include <stdio.h>
#include <conio.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <math.h>



using namespace std;
using namespace cv;


__constant__ float cosine[64];
__constant__ float coef[8];
__constant__ int q_val[64];
__constant__ float cosine2[64];


__global__ void kernel(int *d_a)
{
	__shared__ float arr[8][8];
	__shared__ float out[8][8];
	__shared__ float in[8][8];

	int bx = blockIdx.x;  
	int tx = threadIdx.x; int ty = threadIdx.y;
	arr[ty][tx] = d_a[bx*8 + ty*8 + tx];

	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine[tx + k*8];
		out[ty][k] = in[ty][0]+in[ty][1]+in[ty][2]+in[ty][3]+
						in[ty][4]+in[ty][5]+in[ty][6]+in[ty][7];
		out[ty][k]*=coef[k]; 
	}
	arr[ty][tx] = out[ty][tx];
	for(int k=0;k<8;k++)
	{
		in[ty][tx] = arr[ty][tx]*cosine[ty + k*8];
		out[k][tx] = in[0][tx]+in[1][tx]+in[2][tx]+in[3][tx]+
						in[4][tx]+in[5][tx]+in[6][tx]+in[7][tx];
		out[k][tx]*=coef[k];
	}
	out[ty][tx]/=q_val[ty*8 + tx];
	d_a[bx*8 + ty*8 + tx] = out[ty][tx];

}

__global__ void kernel2(int *d_a)
{
	__shared__ float arr[8][8];
	__shared__ float out[8][8];
	__shared__ float in[8][8];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	arr[ty][tx] = d_a[bx*8 + ty*8 + tx];
	__syncthreads();

	arr[ty][tx]*=q_val[ty*8 + tx];

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

	
	d_a[bx*8+ ty*8 + tx] = (out[ty][tx]);

}


int main(int argc,char**argv)
{ /////////////////////////////Load Image////////////////
	if(argc!=2)
	{
		exit(-2);
	}
	Mat image,im;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image,im,CV_BGR2YCrCb,0);
	int row = (im.rows/8);
	int col
		= (im.cols/8);
	Mat_<Vec3b> _im = im;


	int b_x,b_y,loc;
	int *buf,*d_a;
	float cosval[64],cosval2[64];
	float cons[8];
	cons[0] = sqrt(1.0/8);
	
int quan[64] = 			{	16,	11,     10,     16, 	24,     40,     51,	61,
					12,	12,	14,	19,	26,	58,	60,	55,
					14,	13,	16,	24,	40,	57,	69,	56,
					14,	17,	22,	29,	51,	87,	80,	62,
					18,	22,	37,	56,	68,	109,	103,	77,
					24,	35,	55,	64,	81,	104,	113,	92,
					49,	64,	78,	87,	103,	121,	120,	101,
					72,	92,	95,	98,	112,	100,	103,	99};

	cudaMemcpyToSymbol(q_val,quan,64*sizeof(float));


	cudaError_t er;
	cudaStream_t stream;

	cudaStreamCreate(&stream);
	
	cudaHostAlloc((void**)&buf,64*row*col*sizeof(int),cudaHostAllocDefault);

	er = cudaMalloc((void**)&d_a,256*sizeof(int));
	if(er!=cudaSuccess)
	{
		fprintf(stderr, "failed to allocate device memory\n", cudaGetErrorString(er));
		exit(-1);
	}

	for(int i=1;i<8;i++)
		cons[i] = 0.5;


	for(int i=0;i<8;i++)
		for(int j=0;j<8;j++)
			cosval[i*8 + j] = cos(((2*j + 1)/16.0)*3.14*i);

	cudaMemcpyToSymbol(cosine,cosval,64*sizeof(float));

	for(int i=0;i<8;i++)
		for(int j=0;j<8;j++)
			cosval2[i*8 + j] = cos(((2*i + 1)/16.0)*3.14*j);
	cudaMemcpyToSymbol(cosine2,cosval2,64*sizeof(float));

	cudaMemcpyToSymbol(coef,cons,8*sizeof(float));
	

	for(int b_y=0;b_y<row;b_y++)
	{
		for(b_x=0;b_x<col;b_x++)
		{
			loc = b_y*col + b_x;
			//printf("done till b_y:%d b_x:%d  loc:%d\n",b_y,b_x,loc);
			for(int j=0;j<8;j++)
			{
				for(int i=0;i<8;i++)
				{
					buf[j*8 + i + loc*64] = (int)_im(j + b_y*8,i + b_x*8)[0]; 
				}
			}
		}
	}

		printf("Loaded into buffer\n");

		dim3 dimGrid(4,1,1);
		dim3 dimBlock(8,8);


		for(int i=0;i<64*row*col;i+=256)
		{
			er = cudaMemcpyAsync(d_a,buf + i,256*sizeof(int),cudaMemcpyHostToDevice,stream);
			if(er!=cudaSuccess)
			{
				fprintf(stderr, "failed to transfer to device memory\n", cudaGetErrorString(er));
			}


			kernel<<<dimGrid,dimBlock,0,stream>>>(d_a);
			er = cudaGetLastError();
			if(er!=cudaSuccess)
			{
				fprintf(stderr, "failed to invoke kernel\n", cudaGetErrorString(er));
			
			}

			kernel2<<<dimGrid,dimBlock,0,stream>>>(d_a);
			er = cudaGetLastError();
			if(er!=cudaSuccess)
			{
				fprintf(stderr, "failed to invoke kernel2\n", cudaGetErrorString(er));
			
			}

			er = cudaMemcpyAsync(buf+i,d_a,256*sizeof(int),cudaMemcpyDeviceToHost,stream);
			if(er!=cudaSuccess)
			{
				fprintf(stderr, "failed to transfer back to host\n", cudaGetErrorString(er));
		
			}
			cudaDeviceSynchronize();
			printf("done till %d",i);
		}

		cudaStreamSynchronize(stream);
		cudaDeviceSynchronize();
		cudaStreamDestroy(stream);


		for(int b_y=0;b_y<row;b_y++)
		{
			for(b_x=0;b_x<col;b_x++)
			{
				loc = b_y*col + b_x;
				//printf("done till b_y:%d b_x:%d  loc:%d\n",b_y,b_x,loc);
				for(int j=0;j<8;j++)
				{
					for(int i=0;i<8;i++)
					{
						_im(j + b_y*8,i + b_x*8)[0] = (unsigned char)(buf[j*8 + i + loc*64]); 
					}
				}
			}
		}
		namedWindow("image",CV_WINDOW_NORMAL|CV_WINDOW_KEEPRATIO);
		imshow("image",im);
		cudaFreeHost(buf);
		cudaFree(d_a);
		getch();
		
}


