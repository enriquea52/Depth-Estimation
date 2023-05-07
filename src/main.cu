
#include "computing.cuh"
#include <chrono>
using namespace std::chrono;



template<typename F>
void measureExecutionTime(F f, const std::string& output_file) {
    auto start = high_resolution_clock::now();
    f();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Disparity Computation Took: " << duration.count()/1000000.0 << " Seconds" << std::endl;

    std::stringstream out3d;
    out3d << output_file << ".txt";
    std::ofstream outfile(out3d.str());

    outfile << duration.count()/1000000.0 << " " << "Seconds\n";

}

int main(int argc, const char **argv) {

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE STEREO_MODE[0 or 1] WINDOW_SIZE LAMBDA DMIN" << std::endl;
    std::cerr << "Usage: " <<" LAMBDA is a tunnable parameter for the Dynamic Programming Approach" << std::endl;
    return EXIT_FAILURE;
  }
  // Command Line Arguments Retrieved
  int mode = atoi(argv[4]);
  int window_size = atoi(argv[5]);
  int occlusion = atoi(argv[6]);
  int half_win_size = window_size/2;
  if (window_size % 2 == 0 || window_size < 1) {
  std::cerr << "Please choose an odd number greater or equal than 1\nThe larger the number the larger the computation time" << std::endl;
  return EXIT_FAILURE;
  }
  if (occlusion < 1) {
  std::cerr << "Please provide an occlusion integer value greater than or equal to 1" << std::endl;
  return EXIT_FAILURE;
  }
  if (mode > 2 || mode < 0) {
  std::cerr << "Please select a valid mode: 0 -> Naive Stereo Matching;\n 1 -> Dynamic Programming Stereo " << std::endl;
  return EXIT_FAILURE;
  }

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = atoi(argv[7]);

  // Defining and initializing 2 CV MAT objects
  cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // pixel format uchar ... 8 bit
  cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE); // pixel format uchar ... 8 bit

  // Pointer variables for both host and device computing
  UCHAR * L, * R, * D;
  float * X, * Y, * Z;


  // Output image name initialization
  const std::string output_file = argv[3];

  // Verification of both image1 and image2 
  if (!image1.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }
  if (!image2.data) {
    std::cerr << "No image2 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "window_size = " << window_size << std::endl;
  std::cout << "occlusion_value = " << occlusion << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  // Input Image dimensions 
  int height = image1.size().height; // number of rows
  int width = image1.size().width;   // number of cols
  int size = height*width;           // Total number of pixels of input images

  // Output Image dimensions 
  int height_dis = height - (2*half_win_size); // number of rows
  int width_dis = width - (2*half_win_size);   // number of cols
  int size_dis = height_dis*width_dis;         // Total number of pixels of output images
  int maps_size = width_dis*width_dis;

  // Matrices used for computation of the disparity map using dynamic programming approach
  float *dissim, *C; UCHAR  *M;
  int number_of_rows = 100;     // compute 100 image rows in parallel

  // Memory Allocation for device computing (input and output images)
  cudaMalloc((void**)&L, size*sizeof(UCHAR));
  cudaMalloc((void**)&R, size*sizeof(UCHAR));
  cudaMalloc((void**)&D, size_dis*sizeof(UCHAR));

  // Memory allocation for 3D computed points, for pointcloud visualization
  cudaMalloc((void**)&X, (size_dis)*sizeof(float));cudaMalloc((void**)&Y, (size_dis)*sizeof(float));cudaMalloc((void**)&Z, (size_dis)*sizeof(float));

  // Copy image1/2 data to L/R in device
  cudaMemcpy(L, image1.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);
  cudaMemcpy(R, image2.data, size*sizeof(UCHAR), cudaMemcpyHostToDevice);

  // Naive disparity image
  cv::Mat dpDisparities = cv::Mat::zeros(height_dis, width_dis, CV_8UC1);

  switch (mode)
  {
    case 0:{
      std::cout << "Naive Approach Selected..." << std::endl;
      const auto stereoMatching = std::bind(&naiveApproach, std::cref(L), std::cref(R), std::cref(D), 
                                            width, height, window_size,
                                            baseline, focal_length, dmin);    
      measureExecutionTime(stereoMatching, argv[3]);
    // Copy result C from device to host (c)
    cudaMemcpy(dpDisparities.data, D, size_dis*sizeof(UCHAR), cudaMemcpyDeviceToHost);
    break;
    }
    case 1:{

      // Memory Allocation for device computing (dissimilarity, C and M maps)
      cudaMalloc((void**)&dissim, number_of_rows*maps_size*sizeof(float));
      cudaMalloc((void**)&C, number_of_rows*maps_size*sizeof(float));
      cudaMalloc((void**)&M, number_of_rows*maps_size*sizeof(UCHAR));

      std::cout << "Dynamic Programming Approach Selected..." << std::endl;
      const auto stereoMatching =  std::bind(&dynamicProgrammingApproach, std::cref(L), std::cref(R), std::cref(D), 
                                             std::cref(dissim), std::cref(C), std::cref(M), window_size, number_of_rows, 
                                             width, height, height_dis, width_dis, maps_size, occlusion);
      measureExecutionTime(stereoMatching, argv[3]);
      // Copy result C from device to host (c)
      cudaMemcpy(dpDisparities.data, D, size_dis*sizeof(UCHAR), cudaMemcpyDeviceToHost);      
    break;
    }
    case 2:{

      dpDisparities = cv::Mat::zeros(image1.size().height, image1.size().width, CV_8UC1);
      const auto stereoMatching =  std::bind(&cv_implementation, image1, image2, std::ref(dpDisparities), window_size);
      measureExecutionTime(stereoMatching, argv[3]);
      break;
    }
    default:{
      std::cout << "Please Make sure to select a valid stereo matching mode. 0: naive, 1: DP" << std::endl;
      return 0;
    }
  }


  ////////////
  // Output //
  ////////////

  // save / display images
  std::stringstream out1;
  out1 << output_file << "_dp.png";
  cv::imwrite(out1.str(), dpDisparities);

  // cv::namedWindow("Naive", cv::WINDOW_NORMAL);
  // cv::imshow("Naive", dpDisparities);
  // cv::waitKey(0);

  // Writing pointcloud to .xyz file
  std::cout << "more organized code" << std::endl;
  Disparity2PointCloud(argv[3], height, width, dpDisparities, 
                       window_size, dmin, baseline, focal_length);

  // Free memory 
  cudaFree(L); cudaFree(R); cudaFree(D);
  cudaFree(X); cudaFree(Y); cudaFree(Z);
  cudaFree(dissim); cudaFree(C); cudaFree(M);

  return 0;
}