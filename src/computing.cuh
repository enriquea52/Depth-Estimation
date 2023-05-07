#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <sstream>
#include <string> 
#include <thread>
#include "kernels.cuh"

void dynamicProgrammingApproach(UCHAR *L, UCHAR *R, UCHAR *D, 
                                float *dissim, float *C, UCHAR  *M, 
                                int window_size,int number_of_rows, 
                                int width, int height, int height_dis, 
                                int width_dis, int maps_size, int occlusion);

void naiveApproach(UCHAR *L, UCHAR* R, UCHAR *D, 
                  int width, int height, int win_size,
                  float baseline, float focal_length, int dmin);

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length);


void cv_implementation(cv::Mat& imgL,cv::Mat& imgR, cv::Mat& disparity, int blockSize);
