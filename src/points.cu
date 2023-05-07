
#include "computing.cuh"
#include <chrono>
using namespace std::chrono;

int main(int argc, const char **argv) {

  ///////////////////////////
  // Commandline arguments //
  ///////////////////////////

  ////////////////
  // Parameters //
  ////////////////

  // camera setup parameters
  const double focal_length = 3740;
  const double baseline = 160;

  // stereo estimation parameters
  const int dmin = atoi(argv[2]);

  // Defining and initializing 2 CV MAT objects
  cv::Mat disparities = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // pixel format uchar ... 8 bit


  // Output image name initialization
  const std::string output_file = argv[3];

  // Verification of both image1 and image2 
  if (!disparities.data) {
    std::cerr << "No image1 data" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "------------------ Parameters -------------------" << std::endl;
  std::cout << "focal_length = " << focal_length << std::endl;
  std::cout << "baseline = " << baseline << std::endl;
  std::cout << "disparity added due to image cropping = " << dmin << std::endl;
  std::cout << "output filename = " << argv[3] << std::endl;
  std::cout << "-------------------------------------------------" << std::endl;

  // Input Image dimensions 
  int height = disparities.size().height; // number of rows
  int width = disparities.size().width;   // number of cols
  int size = height*width;           // Total number of pixels of input images


  ////////////
  // Output //
  ////////////

  Disparity2PointCloud(argv[3], height, width, disparities, 
                       1, dmin, baseline, focal_length);


  return 0;
}