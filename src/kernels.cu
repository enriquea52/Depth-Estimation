#include "kernels.cuh"
// Stereo Matching Using Dynamic Programming Optimization

// parallelize the computatio for each pair of rows,
// so the number of threads is equal to height - 2*half_window_size

__global__ 
void naiveStereo(UCHAR *L, UCHAR* R, UCHAR *D, 
                 int width, int height, int win_size,
                 float baseline, float focal_length, int dmin)
{
   int half_win_size = win_size/2;
   int ROW = blockIdx.y*blockDim.y+threadIdx.y; ROW += half_win_size;
   int COL = blockIdx.x*blockDim.x+threadIdx.x; COL += half_win_size;

   if ((ROW < height - half_win_size) && (COL < width - half_win_size))
   {
        int ssd = 0;
        int temp_ssd = 0;
        int min_ssd = INT_MAX;
        int disparity = 0;

        // Iterate over the disparity
        for (int d = -COL + half_win_size; d < width - COL - half_win_size; ++d)
        {
            ssd = 0;
            // iterate over the square window to find the best match
            for (int i = -half_win_size; i <= half_win_size; ++i)
            {
                for (int j = -half_win_size; j <= half_win_size; ++j)
                {
                        temp_ssd =  (L[(ROW + i) * width + (COL+j)] - R[(ROW + i) * width + (COL + d + j)]);
                        ssd += temp_ssd*temp_ssd;
                }
            }
            // if the computed ssd is smaller update
            if (ssd < min_ssd)
            {
                min_ssd = ssd;

                disparity = abs(d);  
            } 
        }
        // Fill respective pixel with the corresponding disparity value
        D[(ROW-half_win_size) * (width-(2*half_win_size)) + (COL-half_win_size)] = disparity;
   }
}


__global__
void dynamicProgramming(float *dissim, float *C, UCHAR *M, UCHAR* D, 
                        int row_num, int width, int height, 
                        int win_size, int maps_size, int occlusion)
{

   int half_window_size = win_size/2;

   int temp_width = width - (2*half_window_size);

   int ROW = blockIdx.x*blockDim.x+threadIdx.x; ROW += half_window_size;

    if (ROW < row_num + half_window_size)
    {
        // Computatinfg the C-map for the given pair of ROWS via Dynamic programming
        CmapComputation(width, half_window_size, ROW, temp_width, maps_size, dissim, C, M, occlusion);

        // trace back from sink and fill disparities
        disparityComputation(width, temp_width, ROW - half_window_size, maps_size, M, D);
    }
    return;
}

__global__
void dissimularityComputation(UCHAR *L, UCHAR* R, UCHAR* D, float *dissim,
                                int width, int height, int row_num, 
                                int win_size, int maps_size)
{

   int half_window_size = win_size/2;

   int temp_width = width - (2*half_window_size);

   int ROW = blockIdx.x*blockDim.x+threadIdx.x; ROW += half_window_size;

   int i = blockIdx.y*blockDim.y+threadIdx.y; i += half_window_size;

   int j = blockIdx.z*blockDim.z+threadIdx.z; j += half_window_size;

   if ((ROW < row_num + half_window_size) && (i < width - half_window_size) && (j < width - half_window_size))
   {

    float i1, i2, sum = 0;

    sum = 0;
    for (int u = -half_window_size; u <= half_window_size; ++u) 
    {
        for (int v = -half_window_size; v <= half_window_size; ++v)
        {
            i1 = static_cast<float>(L[((ROW + v) * width) + (i + u)]);
            i2 = static_cast<float>(R[((ROW + v) * width) + (j + u)]);
            sum += std::abs(i1 - i2); // SAD
        }
    }
    dissim[((i - half_window_size)*temp_width) + (j - half_window_size) + ((ROW-half_window_size)*maps_size)] = sum;

   }
}

__device__
void CmapComputation(int width, int half_window_size, int ROW,
                              int temp_width, int maps_size,
                              float *dissim, float *C,
                              UCHAR *M, int lambda)
{
    float min1, min2, min3, cmin;
    // Initialization of the C map to avoid the first row and column since there is not match in there
    // for(int t = 0; t < temp_width; t++){C[t * temp_width] = 0; C[t] = 0;}

    for (int i = 1; i < temp_width; ++i)     
    {
        for (int j = 1; j < temp_width; ++j) 
        {
            min1 = C[(i-1)*temp_width + (j-1) + ((ROW-half_window_size)*maps_size)] + dissim[i*temp_width + j + ((ROW-half_window_size)*maps_size)];
            min2 = C[(i-1)*temp_width + (j) + ((ROW-half_window_size)*maps_size)] + lambda;
            min3 = C[(i)*temp_width + (j-1) + ((ROW-half_window_size)*maps_size)] + lambda;

            C[i*temp_width + j + ((ROW-half_window_size)*maps_size)] = cmin = fminf(min1, fminf(min2, min3));

            if(min1 == cmin){M[i*temp_width + j + ((ROW-half_window_size)*maps_size)] = 1;}
            if(min2 == cmin){M[i*temp_width + j + ((ROW-half_window_size)*maps_size)] = 2;}
            if(min3 == cmin){M[i*temp_width + j + ((ROW-half_window_size)*maps_size)] = 3;}
        }
    }
    return;
}

__device__
void disparityComputation(int width, int temp_width, int ROW,
                          int maps_size, UCHAR *M, UCHAR *D)
{
    int p, q; // p row, q col
    p = q = temp_width - 1;

    while( p != 0 && q != 0)
    {
        switch (M[p*temp_width + q + (ROW*maps_size)])
        {
        case 1:
            // Setting Best Match
            // multiplied by 3 just for visualization purposes
            D[(ROW*temp_width) + p] = abs(p-q) ; 
            p--; q--;
            break;
        case 2:
        // managing left occlusion
            p--;
            // D[(ROW*temp_width) + p] = 0; 
            D[(ROW*temp_width) + p] = abs(p-q); 
             break;
        case 3:
        // managing right occlusion
            q--;
            // D[(ROW*temp_width) + q] = 0; 
            break;
        default:
            break;
        }
    }

    D[(ROW*temp_width) + p] = abs(p-q); 
    return;
}