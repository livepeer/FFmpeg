#include "cuda/vector_helpers.cuh"

typedef unsigned long long int uint64_cu;

extern "C" {

__global__ void  Subsample_Boxsumint64(cudaTextureObject_t tex,
                                         uint64_cu *dst,
                                         int dst_width, int dst_height, int dst_pitch,
                                         int src_width, int src_height,
                                         int bit_depth)
{

    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        float hscale = (float)src_width / (float)dst_width;
        float vscale = (float)src_height / (float)dst_height;
        
        int xs = (int)(hscale * xo);
        int xe = (int)(xs + hscale); xe = min(src_width-1,xe);
        int ys = (int)(vscale * yo);
        int ye = (int)(ys + vscale); ye = min(src_height-1,ye);

        int index = yo*dst_pitch+xo;
        uint64_cu sum = 0;
        for(int i = xs; i <= xe; i++) {
            for(int j = ys; j <= ye; j++) {
                sum += (uint64_cu)(tex2D<uchar>(tex, i, j));
            }
        }
        dst[index] = sum;
    }
}

}