#include <iostream>

#define THREADS_PER_BLOCK 6

typedef uint8_t UCHAR;

// GPU Device implemented functions

__global__ // Naive Stereo matching approach declaration
void naiveStereo(UCHAR *L, UCHAR* R, UCHAR *D, 
                 int width, int height, int win_size,
                 float baseline, float focal_length, int dmin);

__global__ // Dynamic Programming Stereo matching approach declaration
void dynamicProgramming(float *dissim, float *C, UCHAR *M, UCHAR* D, 
                        int row_num, int width, int height, 
                        int win_size, int maps_size, int occlusion);

__global__ // Computation of the dissimilarity maps
void dissimularityComputation(UCHAR *L, UCHAR* R, UCHAR* D, float *dissim,
                                int width, int height, int row_num, 
                                int win_size, int maps_size);

__device__ // Filling of the C map
void CmapComputation(int width, int half_window_size, int ROW,
                              int temp_width, int maps_size,
                              float *dissim, float *C,
                              UCHAR *M, int lambda);
// lambda value has to go from 0-255 for window size 1


__device__ // Computation of the disparity by finding the shortest path
void disparityComputation(int width, int temp_width, int ROW,
                          int maps_size, UCHAR *M, UCHAR *D);