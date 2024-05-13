

extern "C" __global__ void resize_image_kernel(unsigned char* input_image, unsigned char* output_image, int input_width, int input_height, int output_width, int output_height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        int src_x = x * input_width / output_width;
        int src_y = y * input_height / output_height;

        for (int c = 0; c < channels; ++c) {
            output_image[(y * output_width + x) * channels + c] = input_image[(src_y * input_width + src_x) * channels + c];
        }
    }
}