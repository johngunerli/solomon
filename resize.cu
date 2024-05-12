#include <cuda_runtime.h>

__global__ void resize_kernel(const unsigned char* input_img, unsigned char* output_img, int in_height, int in_width, int out_height, int out_width, float x_ratio, float y_ratio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        int x_mapped = int(x * x_ratio);
        int y_mapped = int(y * y_ratio);

        int output_idx = y * out_width + x;
        int input_idx = y_mapped * in_width + x_mapped;

        output_img[output_idx] = input_img[input_idx];
    }
}
// Since we're calling ctypes in python, we need to use extern "C" to prevent issues.
extern "C" void resize_image(const unsigned char* input_img, unsigned char* output_img, int in_height, int in_width, int out_height, int out_width) {
    float x_ratio = float(in_width) / out_width;
    float y_ratio = float(in_height) / out_height;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, in_height * in_width * sizeof(unsigned char));
    cudaMalloc(&d_output, out_height * out_width * sizeof(unsigned char));

    cudaMemcpy(d_input, input_img, in_height * in_width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    resize_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, in_height, in_width, out_height, out_width, x_ratio, y_ratio);

    cudaMemcpy(output_img, d_output, out_height * out_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
