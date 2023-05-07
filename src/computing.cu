#include "computing.cuh"

void dynamicProgrammingApproach(UCHAR *L, UCHAR *R, UCHAR *D, 
                                float *dissim, float *C, UCHAR  *M, 
                                int window_size,int number_of_rows, 
                                int width, int height, int height_dis, 
                                int width_dis, int maps_size, int occlusion)
{
  // Number of times the disparity computation will take place to cover the whole image 
  // processing a given number of rows in parallel
  int steps = std::ceil(height_dis/(float)number_of_rows);
  std::cout << "steps: " << steps << std::endl;

  int threads = THREADS_PER_BLOCK;
  int block_size = std::ceil(width_dis/threads) + 1;
  int block_size_rows = std::ceil(number_of_rows/threads) + 1;

  const dim3 threadsPerBlock(threads, threads, threads);
  const dim3 blocksPerGrid(block_size_rows, block_size, block_size);

  // Disparity computation (compute n rows t times), image rows = n x t

    for(int iteration = 0; iteration < steps; iteration++)
    {
      std::cout << "Completion Percentage: " << ((iteration+1)/(float)steps) * 100 << std::endl;

      // compute the dissimilarity maps for n rows
      dissimularityComputation<<<blocksPerGrid, threadsPerBlock>>>(L + (number_of_rows*iteration*width),
                                                                   R + (number_of_rows*iteration*width), 
                                                                   D + (number_of_rows*iteration*width_dis), dissim,
                                                                   width, height, number_of_rows, 
                                                                   window_size, maps_size);

      // Apply dynamic programming to n rows to find the n rows of the disparity map
      dynamicProgramming<<<1, number_of_rows>>>(dissim, C, M, D + (number_of_rows*iteration*width_dis), 
                                                number_of_rows, width, height, 
                                                window_size, maps_size, occlusion);
      // Synchronize threads
      cudaDeviceSynchronize();
    }
}

void naiveApproach(UCHAR *L, UCHAR* R, UCHAR *D, 
                 int width, int height, int win_size,
                 float baseline, float focal_length, int dmin)
  {

  int blocksNumber_x = std::ceil(width/(float)THREADS_PER_BLOCK);
  int blocksNumber_y = std::ceil(height/(float)THREADS_PER_BLOCK);

  const dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const dim3 blocksPerGrid(blocksNumber_x, blocksNumber_y);

  naiveStereo<<<blocksPerGrid, threadsPerBlock>>>(L, R, D, 
                                                  width, height, win_size,
                                                  baseline, focal_length, dmin);
  // Synchronize threads
  cudaDeviceSynchronize();
  }

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length)
{
  std::stringstream out3d;
  out3d << output_file << ".xyz";
  std::ofstream outfile(out3d.str());
  #pragma omp parallel for
  for (int i = 0; i < height - window_size; ++i) {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
    for (int j = 0; j < width - window_size; ++j) {
      if (disparities.at<uchar>(i, j) == 0) continue;

      int d = disparities.at<uchar>(i, j) + dmin;
      double u1 = j - (width/2), u2 = u1 + d, v1 = i - (height/2);

      // TODO
      const double Z = (baseline*focal_length)/d;
      const double X = (-baseline)*(u1+u2)/(2*d);
      const double Y = baseline*v1/d;
	  //
      outfile << X/1000.0 << " " << Y/1000.0 << " " << Z/1000.0 << std::endl;
    }
  }
  std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
  std::cout << std::endl;
}

void cv_implementation(cv::Mat& imgL,cv::Mat& imgR, cv::Mat& disparity, int blockSize)
{
  cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
  stereo->setNumDisparities(10 * 16);
  stereo->setBlockSize(blockSize);
  stereo->compute(imgL,imgR,disparity);
  cv::normalize(disparity, disparity, 255, 0, cv::NORM_MINMAX);
	disparity.convertTo(disparity,CV_8U);
}