CUDA_compress
=============
The file kernel.cu contains two main functions kernel and kernel2.kernel is implements the two dimensional Discrete Cosine Transform
(DCT-II)  with some degree of parallelization using the CUDA framework.
The 2D-DCT is implemented as two one dimensional DCT's performed row wise and then column wise.
Each call to the kernel utilizes 64 GPU threads which is much lesser than the 1024 limit but is the only way to parallelize the DCT 
without opting for other algorithms to calculate DCT.The inverse DCT is implemented much the same way in kernel2.

The file image_compress utilizes kernel and kernel2 to find the DCT of 8 by 8 pixel blocks from an image by sequentially loading them into the GPU.The image is loaded using opencv libraries.
Page locked Memory and a stream is used to asynchronously transfer 8 blocks of pixels at a time into GPU global memory and transfer the result back to host memory.

