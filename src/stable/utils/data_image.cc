// STL includes
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <dirent.h>

// C includes
#include <math.h>
#include <assert.h>

// Eigen includes
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>

// PCL includes
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/integral_image_normal.h>

// Local include
#include "data_image.hh"
#include "configuration.hh"
#include "time_stamp.hh"
#include "cv_util.hh"

// OpenCV includes
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DEGREES2RADIANS 0.017453292519943295474f
#define INV_PI 0.31830988618f




using namespace Utils;

inline float sign(float f) {
  return (f >= 0) ? 1 : -1;
}

// Needed to link to the actual static variables.
bool DataImage::m_has_calibration;

float DataImage::m_fx_rgb;
float DataImage::m_fy_rgb;
float DataImage::m_fx_rgb_inv;
float DataImage::m_fy_rgb_inv;
float DataImage::m_cx_rgb;
float DataImage::m_cy_rgb;
float DataImage::m_sequence_width;
float DataImage::m_sequence_height;
bool DataImage::m_is_loaded_from_sequence;


float DataImage::m_fx_depth;
float DataImage::m_fy_depth;
float DataImage::m_fx_depth_inv;
float DataImage::m_fy_depth_inv;
float DataImage::m_cx_depth;
float DataImage::m_cy_depth;

Eigen::Matrix4f DataImage::m_depth_to_rgb_transform;

float DataImage::m_undistort_k1;
float DataImage::m_undistort_k2;
float DataImage::m_undistort_k3;
float DataImage::m_undistort_p1;
float DataImage::m_undistort_p2;


float DataImage::m_baseline_focal;
float DataImage::m_kinect_disparity_offset;
Eigen::Matrix3f DataImage::m_kinect_basic_covariance;

bool DataImage::m_alternative_gradient_distribution_type;
bool DataImage::m_use_rectification_instead_accelerometer;

int DataImage::m_num_normal_neighbors;
int DataImage::m_num_gradient_bins;
bool DataImage::m_is_preprocessed;


//std::vector<std::string> DataImage::m_segmentation_directories;

static  float gauss_mask[5][5] = /* {  {0.0000,    0.0000,    0.0002,    0.0000,    0.0000}  ,
                                                                                                                       {0.0000,    0.0113,    0.0837,    0.0113,    0.0000}  ,
                                                                                                                       {0.0002,    0.0837,    0.6187,    0.0837,    0.0002}  ,
                                                                                                                       {0.0000,    0.0113,    0.0837,    0.0113,    0.0000}  ,
                                                                                                                       {0.0000,    0.0000,    0.0002,    0.0000,    0.0000}    };

                                                                                */


{ {0.0395,    0.0399,    0.0400,    0.0399,    0.0395}  ,
  {0.0399,    0.0403,    0.0404,    0.0403,    0.0399}  ,
  {0.0400,    0.0404,    0.0405,    0.0404,    0.0400}  ,
  {0.0399,    0.0403,    0.0404,    0.0403,    0.0399}  ,
  {0.0395,    0.0399,    0.0400,    0.0399,    0.0395}
};

std::vector<std::string> ListInDir(std::string dirname, std::string extension = std::string("")) {
  std::vector<std::string> result;

  DIR* dp;
  struct dirent* dirp;

  if((dp = opendir(dirname.c_str())) == NULL) {
    std::cerr << "Failed to open folder " << dirname << std::endl;
    exit(EXIT_FAILURE);
  }

  while((dirp = readdir(dp)) != NULL) {
    std::string temp = std::string(dirp->d_name);

    if(extension != "") {
      if(temp.substr(temp.find_last_of(".") + 1) == extension) {
        result.push_back(temp);
      }
    } else {
      result.push_back(temp);
    }
  }

  closedir(dp);
  return result;
}


DataImage::DataImage() {
  m_has_rgb = false;
  m_has_lab = false;
  m_has_annotation = false;
  m_has_depth = false;
  m_has_unary = false;
  m_has_gradient_histogram = false;
  m_has_3d_gradient_histogram = false;
  m_has_3d_features = false;
  m_has_3d_covariance = false;
  m_has_accelerometer_data = false;
  m_has_lab_integral = false;
  m_has_normals = false;
  m_is_initialized = false;
  m_has_point_cloud = false;

  m_clock_cycles = 0;
  m_image_count = 0;

}

void DataImage::SetImageSize(int width, int height){
  if(!m_is_initialized) {
    m_width = width;
    m_height = height;
    m_is_initialized = true;
  } else {
    assert((m_width == width) && (m_height == height));
  }
}

void DataImage::AddRGBImage(const std::string& rgb_image_filename) {
  if(!(rgb_image_filename.substr(rgb_image_filename.find_last_of(".") + 1) == "png")) {
    //Not a png image. This is not supported. Throw an error!
    std::cout << "Only .png are supported. Could not add the image: " << rgb_image_filename << std::endl;
    throw std::runtime_error("Stopping.");
  }

  m_rgb_filename = rgb_image_filename;
  m_rgb_image = cv::imread(rgb_image_filename, CV_LOAD_IMAGE_COLOR);
  m_has_rgb = true;

  SetImageSize(m_rgb_image.cols, m_rgb_image.rows);

  m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(1));
}

void DataImage::AddRGBImage(const cv::Mat& rgb_mat) {
  m_rgb_image = rgb_mat;
  m_has_rgb = true;
  SetImageSize(m_rgb_image.cols, m_rgb_image.rows);

  m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(1));
}


void DataImage::ComputeLABImage(bool store_both) {
  if(m_has_rgb == false) {
    throw std::runtime_error("Cannot compute LAB without loading the color image first.");
  }

  //cv::Mat temp = Graphics::ToOpencvMat(Graphics::RGBToLabCharApproximation(Graphics::Load<unsigned char>(m_rgb_filename)));
  cv::cvtColor(m_rgb_image, m_lab_image, CV_LBGR2Lab);
  m_has_lab = true;

  if(!store_both) {
    m_has_rgb = false;
    m_rgb_image = cv::Mat();
  }
}


void DataImage::AddDepthImage(const std::string& depth_image_filename, bool is_raw) {
  if(depth_image_filename != std::string("")) {
    //Not  needed keeping it for later datasets though.
    //    if (depth_image_filename.substr(depth_image_filename.find_last_of(".") + 1)== "flt") {
    //      std::cout << depth_image_filename << std::endl;
    //      m_depth_image = Graphics::LoadFLT<float>(depth_image_filename, m_baseline_focal, &w, &h);
    //      Image<bool> mask(Width(), m_height,1);
    //      mask.Fill(true);
    //      m_validity_mask = mask;
    //      for(int y=0; y < m_height; y++){
    //        for(int x=0; x < Width(); x++){
    //          float depth = m_depth_image(x,y);
    //          if(m_depth_image(x,y)!= -10.0f || depth > 100){
    //            m_validity_mask(x,y) = false;
    //          }
    //        }
    //      }
    //    }else
    if(depth_image_filename.substr(depth_image_filename.find_last_of(".") + 1) == "pgm") {
//         if(raw_pgm){
//           //Swap the bytes and return them as unsigned short. At least this should be the template.
//           res(x,y) = static_cast<T>(pixBig +(pixSmall << 8 ));
//         }else{
//           //The single precision float in matlab has been multiplied to fit the unsigned short space. Because of this we need to devide by 6553.5 to get the depth value in meters.
//           res(x,y) = static_cast<T>(static_cast<float>(pixSmall +(pixBig << 8 ))/6553.5f);
//           if(res(x,y) ==0){
//             res(x,y) = std::numeric_limits<float>::quiet_NaN();
//           }

      if(is_raw) { //Load the image and flip the bytes then process it.
        cv::Mat depth_image = cv::imread(depth_image_filename, CV_LOAD_IMAGE_ANYDEPTH);
        m_depth_image = ProcessRawDepthImage(depth_image);

        SetImageSize(m_depth_image.cols, m_depth_image.rows);

        m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(0));
        bool* valid;

        for(int y = 0; y < m_height; y++) {
          valid = m_validity_mask.ptr<bool> (y);

          for(int x = 0; x < m_width; x++, valid++) {
            *valid = !isnan(m_depth_image.at<float> (y, x));
          }
        }
      } else { //Load it but do not flip the bytes.
        cv::Mat temp_depth = cv::imread(depth_image_filename, CV_LOAD_IMAGE_ANYDEPTH);

        //As done in my preprocessing for the NYUDepth dataset. Might need to change this for other data?
        temp_depth.convertTo(m_depth_image, CV_32F, 1.0f / 6553.5f, 0);

        //Also set all zero valued pixels to NaN.
        cv::Mat nan_mask;
        cv::inRange(m_depth_image, 0.0f, 0.0f, nan_mask);
        m_depth_image.setTo(std::numeric_limits<float>::quiet_NaN(), nan_mask);
        
        m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(0));
        m_validity_mask.setTo(false, nan_mask);

        SetImageSize(m_depth_image.cols, m_depth_image.rows);

        m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(1));
      }
    } else {
      //Not a supported depth image. Throw an error!
      throw std::runtime_error("Error: Format off depth image is not supporte");
    }

    ComputeDepthNormalization();
    m_has_depth = true;
  }
}

void DataImage::AddDepthImage(cv::Mat& depth_mat) {
  m_depth_image = depth_mat;

  SetImageSize(m_depth_image.cols, m_depth_image.rows);

  m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(0));
  bool* valid;

  for(int y = 0; y < m_height; y++) {
    valid = m_validity_mask.ptr<bool> (y);
    for(int x = 0; x < m_width; x++, valid++) {
      *valid = !isnan(m_depth_image.at<float> (y, x));
    }
  }
  
  ComputeDepthNormalization();
  m_has_depth = true;
}


void DataImage::AddAccelerometerData(const std::string& accel_filename) {
  if(accel_filename != std::string("")) {
    m_accelerometer_rotation = LoadAccelerometerRotation(accel_filename);
    m_has_accelerometer_data = true;
  }
}

void DataImage::AddAccelerometerData(const Eigen::Matrix3f& rotation){
  //Convert the quaternion into a rotation matrix.
  m_accelerometer_rotation = rotation;
  m_has_accelerometer_data = true; 
}

DataImage::~DataImage() {
}

const cv::Mat& DataImage::GetLABImage() const {
  if(!m_has_lab) {
    throw std::runtime_error("No LAB image available");
  }

  if(!m_lab_image.isContinuous()) {
    throw std::runtime_error("LAB not continuous!");
  }

  return m_lab_image;
}

const cv::Mat& DataImage::GetRGBImage() const {
  if(!m_has_rgb) {
    throw std::runtime_error("No RGB image available");
  }

  if(!m_rgb_image.isContinuous()) {
    throw std::runtime_error("RGB not continuous!");
  }

  return m_rgb_image;
}

std::string DataImage::GetRGBFilename() const {
  return m_rgb_filename;
}

cv::Mat const& DataImage::GetDepthImage() const {
  if(!m_has_depth) {
    throw std::runtime_error("No Depth image available");
  }

  return m_depth_image;
}

const cv::Mat& DataImage::GetAnnotation() const {
  if(!m_has_annotation) {
    throw std::runtime_error("No annotation available");
  }

  return m_annotation_image;
}

void DataImage::AddAnnotation(const std::string& filename, const Utils::ColorCoding* cc) {
  if(!(filename.substr(filename.find_last_of(".") + 1) == "png")) {
    throw std::runtime_error("Only .png are supported as annotation images.");
  }

  m_annotation_filename = filename;
  m_annotation_image = cc->BgrToLabel(cv::imread(filename));
  m_has_annotation = true;
}

void DataImage::ComputeColorIntegralImage() {
  //Setup the image
  cv::Mat temp(m_height, m_width, CV_32SC3);

  //Sum up over these to create the integral images.
  for(int y = 0 ; y < m_height; ++y) {
    for(int x = 0 ; x < m_width; ++x) {
      for(int b = 0 ; b < 3; ++b) {
        temp.at<cv::Vec3i> (y, x) [b] = m_lab_image.at<cv::Vec3b> (y, x) [b];

        if(x) {    //if x!=0
          temp.at<cv::Vec3i> (y, x) [b] += temp.at<cv::Vec3i> (y, x - 1) [b];
        }

        if(y) {    //if y!=0
          temp.at<cv::Vec3i> (y, x) [b] += temp.at<cv::Vec3i> (y - 1, x) [b];

          if(x) {    //if x!=0 & y!=0 This is nested as it can never occur if it failed to check for y.
            temp.at<cv::Vec3i> (y, x) [b] -= temp.at<cv::Vec3i> (y - 1, x - 1) [b];
          }
        }
      }
    }
  }

  m_lab_integral_image = temp;
  m_has_lab_integral = true;
}

const std::string& DataImage::GetAnnotationFilename() const {
  return m_annotation_filename;
}

void DataImage::SetAnnotationFilename(const std::string& filename) {
  m_annotation_filename = filename;
}

Eigen::Matrix3f DataImage::GetAccelerometerData() const {
  if(!m_has_accelerometer_data) {
    throw std::runtime_error("No Accelerometer data available");
  }

  return m_accelerometer_rotation;
}

const cv::Mat& DataImage::Get3DFeature() const {
  if(!m_has_3d_features) {
    throw std::runtime_error("No Geometric features available");
  }

  return m_3d_feature;
}

const cv::Mat& DataImage::Get3DCovariance() const {
  if(!m_has_3d_covariance) {
    throw std::runtime_error("No 3D covariance available");
  }

  return m_3d_covariance;
}

void DataImage::AddUnaryPotential(const std::string unary_filename, const float weight) {
  //  std::cout << unary_filename << std::endl;
  if(unary_filename.substr(unary_filename.find_last_of(".") + 1) == "unr") {
    if(m_has_unary) {    // We already have a unary. Add this
      cv::Mat new_unary;
      Utils::ReadMat(unary_filename, new_unary);
      std::runtime_error("Missing implementation! Add this first, sorry! ");

    } else {
      //No unary yet, we use this as the primary unary potential
      Utils::ReadMat(unary_filename, m_unary_potential);
      m_has_unary = true;

      if(weight != 1.0f) {
        for(int y = 0; y < m_height; ++y) {
          float* u = m_unary_potential.ptr<float> (y);

          for(int x = 0; x < m_width; ++x) {
            for(int b = 0; b < m_unary_potential.channels(); ++b, ++u) {
              u[0] *= weight;

              if(u[0] > 1000.0f) {
                u[0] = 1000.0f;
              }

              //      std::cout << u[0] << " ,";
            }

            // std::cout << std::endl;
          }
        }
      }
    }
  } else if(unary_filename.substr(unary_filename.find_last_of(".") + 1) == "dns") {
    throw std::runtime_error("Error: .dns unary files are not implemented yet.");
  } else {
    throw std::runtime_error("Error: unknown unary file format.");
  }
}

void DataImage::AddUnaryPotential(const cv::Mat& unary_potential, const float weight) {
  if(m_has_unary) {    // We already have a unary. Add this
    float temp;

    for(int y = 0; y < m_height; ++y) {
      const float* new_u = unary_potential.ptr<float> (y);
      float* old_u = m_unary_potential.ptr<float> (y);

      for(int x = 0; x < m_width; ++x) {
        for(int b = 0; b < m_unary_potential.channels(); ++b, ++new_u, ++old_u) {
          temp = weight * new_u[0] + old_u[0];
          old_u[0] = (temp > 1000.0f) ? 1000.0f : temp;
        }

      }
    }
  } else { // No unary yet, we use this as the primary unary potential
    m_unary_potential = unary_potential;
    m_has_unary = true;

    if(weight != 1.0f) {
      for(int y = 0; y < m_height; ++y) {
        float* u = m_unary_potential.ptr<float> (y);

        for(int x = 0; x < m_width; ++x) {
          for(int b = 0; b < m_unary_potential.channels(); ++b, ++u) {
            u[0] *= weight;

            if(u[0] > 1000.0f) {
              u[0] = 1000.0f;
            }
          }
        }
      }
    }
  }
}

const cv::Mat& DataImage::GetUnary() const {
  if(!m_has_unary) {
    throw std::runtime_error("No unary available");
  }

  return m_unary_potential;
}

const cv::Mat& DataImage::GetValidityMask() const {
  return m_validity_mask;
}

float DataImage::GetNormalizationFactor(int x_pos) const {
  if(!m_has_depth) {
    throw std::runtime_error("No depth available (caused by call to GetNormalizationFactor(int).");
  }

  return m_normalization_depth_factor[x_pos];
}

const float* DataImage::GetNormalizationData() const {
  if(!m_has_depth) {
    throw std::runtime_error("No depth available (caused by call to GetNormalizationFactor(int).");
  }

  return m_normalization_depth_factor.data();
}

void DataImage::ComputeLabGradientImages() {
  if(!m_has_lab) {
    throw std::runtime_error("Error: Could not compute LAB gradients as no LAB image was available!");
  }

  //Extract L channel
  std::vector<cv::Mat> channels;
  cv::split(m_lab_image, channels);

  //Compute x and y gradients
  cv::Mat sx;
  cv::Sobel(channels[0], sx, CV_32F, 1, 0, 1);
  cv::Mat sy;
  cv::Sobel(channels[0], sy, CV_32F, 0, 1, 1);
  //Compute Magnitude and Orientation
  cv::Mat mag, ori;
  cv::magnitude(sx, sy, mag);
  cv::phase(sx, sy, ori, true);

  //Do the binning
  cv::Mat gradient_hist = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_gradient_bins));


  for(int y = 0; y < mag.rows; ++y)  {
    float* mag_val = mag.ptr<float> (y);
    float* ori_val = ori.ptr<float> (y);
    float* hist = gradient_hist.ptr<float> (y);

    for(int x = 0; x < mag.cols; ++x, ++mag_val, ++ori_val, hist += m_num_gradient_bins)  {
      //Get orientation
      float orientation = ori_val[0];
      float magnitude = mag_val[0];

      //invert if angle >= 180
      if(orientation >= 180) {
        orientation -= 180.0f;
      }

      float b = static_cast<float>(orientation * m_num_gradient_bins) / 180.0f;
      int b0 = b;
      int b1 = b0 + 1;
      float w1 = b - static_cast<float>(b0);
      float w0 = static_cast<float>(b1) - b;
      hist[b0] += w0 * magnitude;
      hist[b1] += w1 * magnitude;
    }
  }

  //Do soft binnning.
  cv::GaussianBlur(gradient_hist, gradient_hist, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

  //Compute the Integral.
  cv::integral(gradient_hist, m_gradient_histogram, CV_32F);



  //  cv::Mat oriMap = cv::Mat::zeros(ori.size(), CV_8UC3);
  //  cv::Vec3b red(0, 0, 1);
  //  cv::Vec3b cyan(1, 1, 0);
  //  cv::Vec3b green(0, 1, 0);
  //  cv::Vec3b yellow(0, 1, 1);
  //  for(int i = 0; i < mag.rows*mag.cols; i++)
  //  {
  //          float* mag_val = reinterpret_cast<float*>(mag.data + i*sizeof(float));
  //          float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
  //          cv::Vec3b* mapPixel = reinterpret_cast<cv::Vec3b*>(oriMap.data + i*3*sizeof(char));
  //          if(*oriPixel < 45.0)
  //            *mapPixel = red* mag_val[0];
  //          else if(*oriPixel >= 45.0 && *oriPixel < 90.0)
  //              *mapPixel = cyan* mag_val[0];
  //          else if(*oriPixel >= 00.0 && *oriPixel < 135.0)
  //              *mapPixel = green* mag_val[0];
  //          else if(*oriPixel >= 135.0 && *oriPixel < 180.0)
  //              *mapPixel = yellow* mag_val[0];
  //  }
  //  imshow("gradient", oriMap);
  //  cv::waitKey(0);


  //  std::vector<cv::Mat> bins;
  //  cv::split(gradient_hist, bins);
  //  for(int b=0; b < m_num_gradient_bins; ++b){
  //    cv::Mat temp(m_height, m_width, CV_8UC1);
  //    for(int y = 0; y < mag.rows; ++y)  {
  //      float* mag_val = bins[b].ptr<float>(y);
  //      uchar* t = temp.ptr<uchar>(y);
  //      for(int x = 0; x < mag.cols; ++x, ++mag_val, ++t)  {
  //        t[0] = mag_val[0];
  //      }
  //    }
  //    std::stringstream name;
  //    name << "bin" << b;
  //    cv::imshow(name.str().c_str(), temp);

  //  }
  //  cv::waitKey(0);


  //  // Setup the images to store the data in.
  //  Image<float> temp(m_width, m_height,m_num_gradient_bins);
  //  temp.Fill(0.0f);
  //  Image<float> oriented_magnitude(m_width, m_height,2);
  //  temp.Fill(0.0f);


  //  // Compute the per pixel gradient and store it's orientation and magnitude
  //  for (int j = 0; j < m_height; j++) {
  //    int j1 = j+1 < m_height ? j+1 : j;
  //    int j2 = j > 0 ? j-1 : j;
  //    for (int i = 0; i < m_width; i++) {
  //      int i1 = i+1 < m_width ? i+1 : i;
  //      int i2 = i > 0 ? i-1 : i;
  //      float dx = (m_lab_image.at<cv::Vec3b>(j, i1)[c] - m_lab_image.at<cv::Vec3b>(j, i2)[c]) / (i1-i2);
  //      float dy = (m_lab_image.at<cv::Vec3b>(j1, i)[c] - m_lab_image.at<cv::Vec3b>(j2, i)[c]) / (j1-j2);
  //      oriented_magnitude(i,j,0)  = atan2(dx, dy);
  //      oriented_magnitude(i,j,1)  = sqrt(dx*dx + dy*dy);
  //    }
  //  }

  //  //Take a window around each pixel and do soft binning.
  //  for (int j = 0; j < m_height; j++) {
  //    int j2 = j+2 < m_height ? j+2 : m_height-1;
  //    int j1 = j-2 >= 0 ? j-2 : 0;
  //    for (int i = 0; i < m_width; i++) {
  //      int i2 = i+2 < m_width ? i+2 : m_width-1;
  //      int i1 = i-2 >= 0 ? i-2 : 0;
  //      float a = oriented_magnitude(i,j,0);
  //      float m = oriented_magnitude(i,j,1);
  //      // Signed binning
  //      if (a < 0) {
  //        a += M_PI;
  //        m = -m;                                                     //// TODO: CHECK THIS!!!
  //      }
  //      a = m_num_gradient_bins * a * INV_PI;
  //      if (a >= m_num_gradient_bins) {
  //        a = m_num_gradient_bins - 1;
  //      }
  //      // Linear interpolation
  //      int a1 = a;
  //      int a2 = a + 1;
  //      float w1 = a2 - a;
  //      float w2 = a - a1;
  //      if (a2 >= m_num_gradient_bins) {
  //        a2 = 0;
  //      }


  //      //Calculate the soft bin count.
  //      for(int y=j1; y <=j2; ++y){
  //        for(int x=i1; x <=i2; ++x){
  //          temp(x,y,a1) += w1*m* gauss_mask[x-i1][y-j1];
  //          temp(x,y,a2) += w2*m* gauss_mask[x-i1][y-j1];
  //        }
  //      }

  //      //		temp(i,j,a1) = w1*m;
  //      //		temp(i,j,a2) = w2*m;
  //    }
  //  }
  //======================begin================================
  //Do the max pooling
  /*  Image<float> temp2(m_width, m_height,m_num_gradient_bins);
  for (int j = 0; j < m_height; j++) {
    int j2 = j+2 < m_height ? j+2 : m_height-1;
    int j1 = j-2 >= 0 ? j-2 : 0;
    for (int i = 0; i < m_width; i++) {
      int i2 = i+2 < m_width ? i+2 : m_width-1;
      int i1 = i-2 >= 0 ? i-2 : 0;


        //Calculate the soft bin count.
        for(int y=j1; y <=j2; ++y){
          for(int x=i1; x <=i2; ++x){
            temp(x,y,a1) += w1*m* gauss_mask[x-i1][y-j1];
            temp(x,y,a2) += w2*m* gauss_mask[x-i1][y-j1];
          }
        }

        //		temp(i,j,a1) = w1*m;
        //		temp(i,j,a2) = w2*m;
      }
    }

  }
  temp = temp2;*/
  //======================end================================

  //Sum up over these to create the integral images.
  //  Image<float> temp2(m_width+1, m_height+1, m_num_gradient_bins);
  //  temp2.Fill(0.0f);
  //  for(int y=1 ; y < m_height+1; ++y){
  //    for(int x=1 ; x < m_width+1; ++x){
  //      for(int b=0 ; b < temp.Bands(); ++b){
  //        temp2(x,y,b) = temp(x-1,y-1,b);
  //        temp2(x,y,b) += temp2(x-1,y,b);
  //        temp2(x,y,b) += temp2(x,y-1,b);
  //        temp2(x,y,b) -= temp2(x-1,y-1,b);

  //      }
  //    }
  //  }
  // m_gradient_histogram = Graphics::ToOpencvMat(temp2);
  m_has_gradient_histogram = true;
}

void DataImage::ComputeDepthGradientImages() {
  if(!m_has_depth) {
    throw std::runtime_error("Error: Could not compute Depth gradients as no depth was loaded!");
  }

  //Compute x and y gradients in the depth image
  cv::Mat sx;
  cv::Sobel(m_depth_image, sx, CV_32F, 1, 0, 1);
  cv::Mat sy;
  cv::Sobel(m_depth_image, sy, CV_32F, 0, 1, 1);
  //Compute Magnitude and Orientation
  cv::Mat mag, ori;
  cv::magnitude(sx, sy, mag);
  cv::phase(sx, sy, ori, true);

  //Do the binning
  cv::Mat gradient_hist = cv::Mat::zeros(m_height, m_width, CV_32FC(m_num_gradient_bins));

  for(int y = 0; y < mag.rows; ++y)  {
    float* mag_val = mag.ptr<float> (y);
    float* ori_val = ori.ptr<float> (y);
    float* hist = gradient_hist.ptr<float> (y);

    for(int x = 0; x < mag.cols; ++x, ++mag_val, ++ori_val, hist += m_num_gradient_bins)  {
      //Get orientation
      float orientation = ori_val[0];
      float magnitude = mag_val[0];

      //Make sure we have no NaN values!
      if(!(orientation == orientation)) {
        continue; //Just skip this pixel if it doesn't have a gradient!
      }

      //invert if angle >= 180
      if(orientation >= 180) {
        orientation -= 180.0f;
      }

      float b = static_cast<float>(orientation * m_num_gradient_bins) / 180.0f;
      int b0 = b;
      int b1 = b0 + 1;
      float w1 = b - static_cast<float>(b0);
      float w0 = static_cast<float>(b1) - b;
      hist[b0] += w0 * magnitude;
      hist[b1] += w1 * magnitude;
    }
  }

  //Do soft binnning.
  cv::GaussianBlur(gradient_hist, gradient_hist, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

  //Compute the Integral.
  cv::integral(gradient_hist, m_gradient_3d_histogram, CV_32F);


//  // Setup the images to store the data in.
//  Image<float> temp(m_width, m_height,m_num_gradient_bins);
//  temp.Fill(0.0f);
//  Image<float> oriented_magnitude(m_width, m_height,2);
//  temp.Fill(0.0f);

//  // Compute the per pixel gradient and store it's orientation and magnitude
//  for (int j = 0; j < m_height; j++) {
//    int j1 = j+1 < m_height ? j+1 : j;
//    int j2 = j > 0 ? j-1 : j;
//    for (int i = 0; i < m_width; i++) {
//      int i1 = i+1 < m_width ? i+1 : i;
//      int i2 = i > 0 ? i-1 : i;
//      float dx = (m_depth_image.at<float>(j, i1) - m_depth_image.at<float>(j, i2)) / (i1-i2);
//      float dy = (m_depth_image.at<float>(j1, i) - m_depth_image.at<float>(j2, i)) / (j1-j2);
//      if(isnan(dx)|| isnan(dy)){
//        dx=0;
//        dy=0;  //TODO ... dirty fix, maybe change at some point or use some validity mask here as well ?
//      }
//      oriented_magnitude(i,j,0)  = atan2(dx, dy);
//      oriented_magnitude(i,j,1)  = sqrt(dx*dx + dy*dy);
//    }
//  }

//  //Take a window around each pixel and do soft binning.
//  for (int j = 0; j < m_height; j++) {
//    int j2 = j+2 < m_height ? j+2 : m_height-1;
//    int j1 = j-2 >= 0 ? j-2 : 0;
//    for (int i = 0; i < m_width; i++) {
//      int i2 = i+2 < m_width ? i+2 : m_width-1;
//      int i1 = i-2 >= 0 ? i-2 : 0;
//      float a = oriented_magnitude(i,j,0);
//      float m = oriented_magnitude(i,j,1);
//      // Signed binning
//      if (a < 0) {
//        a += M_PI;
//        m = -m;                                                     //// TODO: CHECK THIS!!!
//      }
//      a = m_num_gradient_bins * a * INV_PI;
//      if (a >= m_num_gradient_bins) {
//        a = m_num_gradient_bins - 1;
//      }
//      // Linear interpolation
//      int a1 = a;
//      int a2 = a + 1;
//      float w1 = a2 - a;
//      float w2 = a - a1;
//      if (a2 >= m_num_gradient_bins) {
//        a2 = 0;
//      }
//      //Calculate the soft bin count.
//      for(int y=j1; y <=j2; ++y){
//        for(int x=i1; x <=i2; ++x){
//          temp(x,y,a1) += w1*m* gauss_mask[x-i1][y-j1];
//          temp(x,y,a2) += w2*m* gauss_mask[x-i1][y-j1];
//        }
//      }
//    }
//  }
//  //Sum up over these to create the integral images.
//  Image<float> temp2(m_width+1, m_height+1, m_num_gradient_bins);
//  temp2.Fill(0.0f);
//  for(int y=1 ; y < m_height+1; ++y){
//    for(int x=1 ; x < m_width+1; ++x){
//      for(int b=0 ; b < temp.Bands(); ++b){
//        temp2(x,y,b) = temp(x-1,y-1,b);
//        temp2(x,y,b) += temp2(x-1,y,b);
//        temp2(x,y,b) += temp2(x,y-1,b);
//        temp2(x,y,b) -= temp2(x-1,y-1,b);
//
//      }
//    }
//  }

//  m_gradient_3d_histogram = Graphics::ToOpencvMat(temp2);
  m_has_3d_gradient_histogram = true;
}


float DataImage::GetGradientPatch(int x, int y, int gradient_band, int patch_radius) const {
  assert(m_has_gradient_histogram);
  int x1 = x - patch_radius;
  int x2 = x + patch_radius;
  int y1 = y - patch_radius;
  int y2 = y + patch_radius;

  if(x1 < 0) {
    x1 = 0;
  }

  if(x2 >= m_width) {
    x2 = m_width - 1;
  }

  if(y1 < 0) {
    y1 = 0;
  }

  if(y2 >= m_height) {
    y2 = m_height - 1;
  }

  const float* data = m_gradient_histogram.ptr<float> (0);

  float value = data[m_num_gradient_bins * (x2 + 1 + (y2 + 1) * (m_width + 1)) + gradient_band];

  value -=  data[m_num_gradient_bins * (x1 + (y2 + 1) * (m_width + 1)) + gradient_band];

  value -=  data[m_num_gradient_bins * (x2 + 1 + y1 * (m_width + 1)) + gradient_band];

  value +=  data[m_num_gradient_bins * (x1 + y1 * (m_width + 1)) + gradient_band];


  //Scale up the values if the patch is at a border.
  float patch_size = (2 * patch_radius + 1) * (2 * patch_radius + 1);

  float actual_patch_size = (x2 - x1 + 1) * (y2 - y1 + 1);

  if(patch_size > actual_patch_size) {
    return patch_size / actual_patch_size * value;
  } else {
    return value;
  }

}


float DataImage::GetColorPatch(int x, int y, int color_channel, int patch_width, int patch_height) const {
  assert(m_has_lab_integral);
  int border = 10;
  int forced_size = 3;
  int x1 = x - patch_width;
  int x2 = x + patch_width;
  int y1 = y - patch_height;
  int y2 = y + patch_height;

  if(x1 < border) {
    x1 = border;

    if(x2 < x1 + forced_size) {
      x2 = x1 + forced_size;
    }
  }

  if(x2 >= m_width - border) {
    x2 = m_width - 1 - border;

    if(x1 >= x2 - forced_size) {    // can only be true if x2 is > width -1 -border.
      x1 = x2 - forced_size;
    }
  }

  if(y1 < border) {
    y1 = border;

    if(y2 < y1 + forced_size) {    // can only be true if y1 is < border.
      y2 = y1 + forced_size;
    }
  }

  if(y2 >= m_height - border) {
    y2 = m_height - 1 - border;

    if(y1 >= y2 - forced_size) {    // can only be true if y2 is > height -1 -border.
      y1 = y2 - forced_size;
    }
  }

  float value = m_lab_integral_image.at<cv::Vec3i> (y2, x2) [color_channel];

  if(x1 > 0) {
    value -=  m_lab_integral_image.at<cv::Vec3i> (y2, x1 - 1) [color_channel];
  }

  if(y1 > 0) {
    value -=  m_lab_integral_image.at<cv::Vec3i> (y1 - 1, x2) [color_channel];

    if(x1 > 0) {
      value +=  m_lab_integral_image.at<cv::Vec3i> (y1 - 1, x1 - 1) [color_channel];
    }
  }

  //Scale up the values if the patch is at a border.
  float patch_size = (2 * patch_width + 1) * (2 * patch_height + 1);
  float actual_patch_size = (x2 - x1 + 1) * (y2 - y1 + 1);

  if(patch_size > actual_patch_size) {
    return patch_size / actual_patch_size * value;
  } else {
    return value;
  }

}


float DataImage::GetGradientPatch(int x, int y, int gradient_band, int patch_width, int patch_height) const {
  assert(m_has_gradient_histogram);
  // int border =10;
  // int forced_size = 3;
  int x1 = x - patch_width;
  int x2 = x + patch_width;
  int y1 = y - patch_height;
  int y2 = y + patch_height;
  /* if(x1 < border){
    x1=border;
    if(x2 < x1 +forced_size){
      x2=x1 + forced_size;
    }
  }
  if(x2 >= m_width- border) {
    x2= m_width -1 - border;
    if(x1 >= x2 - forced_size) { // can only be true if x2 is > width -1 -border.
      x1= x2 - forced_size;
    }
  }

  if(y1 < border){
    y1=border;
    if(y2 < y1 +forced_size){ // can only be true if y1 is < border.
      y2= y1 + forced_size;
    }
  }

  if(y2 >= m_height- border) {
    y2= m_height -1 - border;
    if(y1 >= y2 - forced_size) { // can only be true if y2 is > height -1 -border.
      y1= y2 - forced_size;
    }
  }
  */
  x1 = std::max(0, x1);
  x1 = std::min(x1, m_width - 1);
  y1 = std::max(0, y1);
  y1 = std::min(y1, m_height - 1);
  x2 = std::max(0, x2);
  x2 = std::min(x2, m_width - 1);
  y2 = std::max(0, y2);
  y2 = std::min(y2, m_height - 1);

  const float* data = m_gradient_histogram.ptr<float> (0);
  float value = data[m_num_gradient_bins * (x2 + 1 + (y2 + 1) * (m_width + 1)) + gradient_band];
  value -=  data[m_num_gradient_bins * (x1 + (y2 + 1) * (m_width + 1)) + gradient_band];
  value -=  data[m_num_gradient_bins * (x2 + 1 + y1 * (m_width + 1)) + gradient_band];
  value +=  data[m_num_gradient_bins * (x1 + y1 * (m_width + 1)) + gradient_band];


  //Scale up the values if the patch is at a border.
  //float patch_size = (2*patch_width +1) * ( 2*patch_height +1);
  float actual_patch_size = (x2 - x1 + 1) * (y2 - y1 + 1);

  /* if(patch_size > actual_patch_size ){
    return patch_size / actual_patch_size * value;
  }else{
    return value;
  }*/
  return value / actual_patch_size;

}

const cv::Mat& DataImage::GetGradient3dData() const {
  if(!m_has_3d_gradient_histogram) {
    throw std::runtime_error("Depth gradient data was requested but it was not stored for a data image! Check your load flags please :)");
  }

  return m_gradient_3d_histogram;
}

const cv::Mat& DataImage::GetGradientData() const {
  if(!m_has_gradient_histogram) {
    throw std::runtime_error("Color gradient data was requested but it was not stored for a data image! Check your load flags please :)");
  }

  return m_gradient_histogram;
}

const cv::Mat& DataImage::GetColorIntegralData() const {
  if(!m_has_lab_integral) {
    throw std::runtime_error("Color integral data was requested but it was not stored for a data image! Check your load flags please :)");
  }

  return m_lab_integral_image;
}

const cv::Mat& DataImage::GetNormals() const {
  if(!m_has_normals) {
    throw std::runtime_error("Normal data was requested but it was not stored for a data image! Check your load flags please :)");
  }

  return m_normals;
}

float DataImage::GetGradient3dPatch(int x, int y, int gradient_band, int patch_width, int patch_height) const {
  assert(m_has_3d_gradient_histogram);
  // int border =10;
  // int forced_size = 3;
  int x1 = x - patch_width;
  int x2 = x + patch_width;
  int y1 = y - patch_height;
  int y2 = y + patch_height;
  /* if(x1 < border){
    x1=border;
    if(x2 < x1 +forced_size){
      x2=x1 + forced_size;
    }
  }
  if(x2 >= m_width- border) {
    x2= m_width -1 - border;
    if(x1 >= x2 - forced_size) { // can only be true if x2 is > width -1 -border.
      x1= x2 - forced_size;
    }
  }

  if(y1 < border){
    y1=border;
    if(y2 < y1 +forced_size){ // can only be true if y1 is < border.
      y2= y1 + forced_size;
    }
  }

  if(y2 >= m_height- border) {
    y2= m_height -1 - border;
    if(y1 >= y2 - forced_size) { // can only be true if y2 is > height -1 -border.
      y1= y2 - forced_size;
    }
  }
  */
  x1 = std::max(0, x1);
  x1 = std::min(x1, m_width - 1);
  y1 = std::max(0, y1);
  y1 = std::min(y1, m_height - 1);
  x2 = std::max(0, x2);
  x2 = std::min(x2, m_width - 1);
  y2 = std::max(0, y2);
  y2 = std::min(y2, m_height - 1);


  const float* data = m_gradient_3d_histogram.ptr<float> (0);
  float value = data[m_num_gradient_bins * (x2 + 1 + (y2 + 1) * (m_width + 1)) + gradient_band];
  value -=  data[m_num_gradient_bins * (x1 + (y2 + 1) * (m_width + 1)) + gradient_band];
  value -=  data[m_num_gradient_bins * (x2 + 1 + y1 * (m_width + 1)) + gradient_band];
  value +=  data[m_num_gradient_bins * (x1 + y1 * (m_width + 1)) + gradient_band];

  //Scale up the values if the patch is at a border.
  //float patch_size = (2*patch_width +1) * ( 2*patch_height +1);
  float actual_patch_size = (x2 - x1 + 1) * (y2 - y1 + 1);

  /* if(patch_size > actual_patch_size ){
    return patch_size / actual_patch_size * value;
  }else{
    return value;
  }*/
  return value / actual_patch_size;
}

//void DataImage::SaveGradientImages(std::string filename, bool from_depth) const{
//  for(int b = 0; b < m_gradient_histogram.Bands(); ++b){
//    //Create a new image to save in.
//    Image<unsigned char> result(m_gradient_histogram.Width(), m_gradient_histogram.Height(), 3);

//    //loop over the image.
//    unsigned char value;
//    for(int y= 0 ; y < result.Height(); ++y){
//      for(int x= 0 ; x < result.Width(); ++x){
//        if(from_depth){
//          value = (GetGradientPatch(x,y,b,0)+  10.0f ) * 255.0f / 20.0f;
//        }else{
//          value = (GetGradient3dPatch(x,y,b,1, 1)+  10.0f ) * 255.0f / 20.0f;
//        }
//        result(x,y,0) = value;
//        result(x,y,1) = value;
//        result(x,y,2) = value;
//      }
//    }
//    std::stringstream s;
//    s << filename << b <<".png";
//    Save<unsigned char>(result, s.str());
//  }
//}

void DataImage::LoadAndUndistortDepthPlusRGB(const std::string& rgb_filename, const std::string& depth_filename, const std::string& accelerometer_filename) {
  if(!m_has_calibration){
    std::runtime_error("Error: Trying to use a calibration while it is not set.");
  }
  AddRGBImage(rgb_filename);
  AddDepthImage(depth_filename, true);
  AddAccelerometerData(accelerometer_filename);


  cv::Mat temp_depth(m_height, m_width, CV_32FC1);
  temp_depth.setTo(std::numeric_limits<float>::quiet_NaN());
  //Image<unsigned char> temp_rgb(width, height, 3);
  cv::Mat temp_rgb(m_height, m_width, CV_8UC3, cv::Scalar::all(255));


  //Fill the validity mask with false; Set it to true later on if a pixel is assigned both to the rgb and depth point.
  m_validity_mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar::all(0));

  float x_r, y_r, r2, r4, r6, cdist, xd1, yd1, a1, a2, a3, deltax, deltay, xd_res, yd_res, inter_x, inter_y;
  int x0, y0, x1, y1;
  bool xl, xr, yt, yb;

  //Loop over both images in one pass and undistorts them.
  bool* valid;

  for(int y = 0 ; y < m_height; ++y) {
    valid = m_validity_mask.ptr<bool> (y);

    for(int x = 0 ; x < m_width; ++x, ++valid) {
      x_r = (static_cast<float>(x) - m_cx_rgb) * m_fx_rgb_inv;
      y_r = (static_cast<float>(y) - m_cy_rgb) * m_fy_rgb_inv;

      r2 = x_r * x_r + y_r * y_r;
      r4 = r2 * r2;
      r6 = r2 * r4;

      cdist = 1 + m_undistort_k1 * r2 + m_undistort_k2 * r4 + m_undistort_k3 * r6;
      xd1 = x_r * cdist;
      yd1 = y_r * cdist;

      a1 = 2 * x_r * y_r;
      a2 = r2 + 2 * x_r * x_r;
      a3 = r2 + 2 * y_r * y_r;
      deltax = m_undistort_p1 * a1 + m_undistort_p2 * a2;
      deltay = m_undistort_p1 * a3 + m_undistort_p2 * a1;

      xd_res = xd1 + deltax;
      yd_res = yd1 + deltay;

      xd_res = m_fx_rgb * xd_res + m_cx_rgb;
      yd_res = m_fy_rgb * yd_res + m_cy_rgb;

      //Interpolate between pixels. first calculate 4 locations.

      x0 = floor(xd_res);
      y0 = floor(yd_res);
      x1 = x0 + 1;
      y1 = y0 + 1;

      //for the rgb, just check which positions are in the image and interpolate.

      //then check if the values are valid
      xl = (x0 >= 0) && (x0 < m_width);
      xr = (x1 >= 0) && (x1 < m_width);
      yt = (y0 >= 0) && (y0 < m_height);
      yb = (y1 >= 0) && (y1 < m_height);

      if((!xl && !xr) || (!yt && !yb)) {
        //non of the pixels are valid for interpolation. We leave this pixel as it is.
        continue;
      }


      //Start interpolating
      inter_x = xd_res - static_cast<float>(x0);
      inter_y = yd_res - static_cast<float>(y0);

      //First interpolate the upper two and the lower two pixels.
      float top_r = 0;
      float top_g = 0;
      float top_b = 0;
      float top_d = 0;
      bool upper = true;
      bool upper_depth = true;


      if(xl && yt) {
        if(xr && yt) {    //both upper are valid.
          top_r = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x0) [0]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x1) [0]);
          top_g = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x0) [1]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x1) [1]);
          top_b = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x0) [2]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y0, x1) [2]);

          if(!isnan(m_depth_image.at<float> (y0, x0))) {
            if(!isnan(m_depth_image.at<float> (y0, x1))) {         // both are okay.
              top_d = (1.0f - inter_x) *  m_depth_image.at<float> (y0, x0) + (inter_x) *  m_depth_image.at<float> (y0, x1);
            } else { //left is okay.
              top_d = m_depth_image.at<float> (y0, x0);
            }
          } else if(!isnan(m_depth_image.at<float> (y0, x1))) {         // right is okay
            top_d = m_depth_image.at<float> (y0, x1);
          } else { //both are nan.
            upper_depth = false;
          }
        } else { //only left upper is valid.
          top_r = m_rgb_image.at<cv::Vec3b> (y0, x0) [0];
          top_g = m_rgb_image.at<cv::Vec3b> (y0, x0) [1];
          top_b = m_rgb_image.at<cv::Vec3b> (y0, x0) [2];

          if(!isnan(m_depth_image.at<float> (y0, x0))) {
            top_d = m_depth_image.at<float> (y0, x0);
          } else {
            upper_depth = false;
          }
        }

      } else if(xr && yt) {    // only right upper is valid.
        top_r = m_rgb_image.at<cv::Vec3b> (y0, x1) [0];
        top_g = m_rgb_image.at<cv::Vec3b> (y0, x1) [1];
        top_b = m_rgb_image.at<cv::Vec3b> (y0, x1) [2];

        if(!isnan(m_depth_image.at<float> (y0, x1))) {
          top_d = m_depth_image.at<float> (y0, x1);
        } else {
          upper_depth = false;
        }

      } else { // only check lower values for interpolation.
        upper = false;
        upper_depth = false;
      }

      float bottom_r = 0;
      float bottom_g = 0;
      float bottom_b = 0;
      float bottom_d = 0;
      bool lower = true;
      bool lower_depth = true;

      if(xl && yb) {
        if(xr && yb) {    //both upper are valid.
          bottom_r = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x0) [0]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x1) [0]);
          bottom_g = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x0) [1]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x1) [1]);
          bottom_b = (1.0f - inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x0) [2]) + (inter_x) * static_cast<float>(m_rgb_image.at<cv::Vec3b>(y1, x1) [2]);

          if(!isnan(m_depth_image.at<float> (y1, x0))) {
            if(!isnan(m_depth_image.at<float> (y1, x1))) {         // both are okay.
              bottom_d = (1.0f - inter_x) *  m_depth_image.at<float> (y1, x0) + (inter_x) *  m_depth_image.at<float> (y1, x1);
            } else { //left is okay.
              bottom_d = m_depth_image.at<float> (y1, x0);
            }
          } else if(!isnan(m_depth_image.at<float> (y1, x1))) {         // right is okay
            bottom_d = m_depth_image.at<float> (y1, x1);
          } else { //both are nan.
            lower_depth = false;
          }

        } else { //only left upper is valid.
          bottom_r = m_rgb_image.at<cv::Vec3b> (y1, x0) [0];
          bottom_g = m_rgb_image.at<cv::Vec3b> (y1, x0) [1];
          bottom_b = m_rgb_image.at<cv::Vec3b> (y1, x0) [2];

          if(!isnan(m_depth_image.at<float> (y1, x0))) {
            bottom_d = m_depth_image.at<float> (y1, x0);
          } else {
            lower_depth = false;
          }
        }

      } else if(xr && yb) {    // only right upper is valid.
        bottom_r = m_rgb_image.at<cv::Vec3b> (y1, x1) [0];
        bottom_g = m_rgb_image.at<cv::Vec3b> (y1, x1) [1];
        bottom_b = m_rgb_image.at<cv::Vec3b> (y1, x1) [2];

        if(!isnan(m_depth_image.at<float> (y1, x1))) {
          bottom_d = m_depth_image.at<float> (y1, x1);
        } else {
          lower_depth = false;
        }

      } else { // only check lower values for interpolation.
        lower = false;
        lower_depth = false;
      }

      if(lower) {
        if(upper) {    //we can use both.
          temp_rgb.at<cv::Vec3b> (y, x) [0] = static_cast<unsigned char>(top_r * (1.0f - inter_y) + bottom_r * (inter_y));
          temp_rgb.at<cv::Vec3b> (y, x) [1] = static_cast<unsigned char>(top_g * (1.0f - inter_y) + bottom_g * (inter_y));
          temp_rgb.at<cv::Vec3b> (y, x) [2] = static_cast<unsigned char>(top_b * (1.0f - inter_y) + bottom_b * (inter_y));
        } else { //we can only use the lower data
          temp_rgb.at<cv::Vec3b> (y, x) [0] = static_cast<unsigned char>(bottom_r);
          temp_rgb.at<cv::Vec3b> (y, x) [1] = static_cast<unsigned char>(bottom_g);
          temp_rgb.at<cv::Vec3b> (y, x) [2] = static_cast<unsigned char>(bottom_b);
        }
      } else if(upper) {    //we can only use the upper data.
        temp_rgb.at<cv::Vec3b> (y, x) [0] = static_cast<unsigned char>(top_r);
        temp_rgb.at<cv::Vec3b> (y, x) [1] = static_cast<unsigned char>(top_g);
        temp_rgb.at<cv::Vec3b> (y, x) [2] = static_cast<unsigned char>(top_b);
      } else {
        std::cerr << "This point should never be reached! RGB";
      }

      if(lower_depth) {
        if(upper_depth) {    //we can use both.
          temp_depth.at<float> (y, x) = top_d * (1.0f - inter_y) + bottom_d * (inter_y);
          *valid = true;
        } else { //we can only use the lower data
          temp_depth.at<float> (y, x) = bottom_d;
          *valid = true;
        }
      } else if(upper_depth) {    //we can only use the upper data.
        temp_depth.at<float> (y, x) = top_d;
        *valid = true;
      } else {
        //       std::cerr << "No depth at this pixel :( ";
      }
    }
  }

  m_depth_image = temp_depth;
  m_has_depth = true;
  m_rgb_image = temp_rgb;
  m_has_rgb = true;
}

void DataImage::SetRGBIntrinsic(std::string rgb_camera_calibration_filename) {
  std::ifstream in(rgb_camera_calibration_filename.c_str());

  if(!in) {
    std::cerr << "Cannot open file " << rgb_camera_calibration_filename << std::endl;
    return;
  }

  float tmp;
  in >> tmp;
  m_fx_rgb = tmp;//(0,0)
  in >> tmp; //(0,1);
  in >> tmp;
  m_cx_rgb = tmp;//(0,2);
  in >> tmp; //(1,0);
  in >> tmp;
  m_fy_rgb = tmp;//(1,1);
  in >> tmp;
  m_cy_rgb = tmp;//(1,2);
  m_fx_rgb_inv = 1.0f / m_fx_rgb;
  m_fy_rgb_inv = 1.0f / m_fy_rgb;

  //Ship the next entries. as we do not care about them currently.
  for(int i = 0; i < 19; ++i) {
    in >> tmp;
  }

  //Take the next five and set the undistortion parameters.
  in >> tmp;
  m_undistort_k1 = tmp;
  in >> tmp;
  m_undistort_k2 = tmp;
  in >> tmp;
  m_undistort_k3 = tmp;
  in >> tmp;
  m_undistort_p1 = tmp;
  in >> tmp;
  m_undistort_p2 = tmp;
  in.close();
}

void DataImage::SetRGBAndDepthIntrinsic(std::string rgb_camera_calibration_filename, std::string depth_camera_calibration_filename) {

  SetRGBIntrinsic(rgb_camera_calibration_filename);

  std::ifstream in(depth_camera_calibration_filename.c_str());

  if(!in) {
    std::cerr << "Cannot open file " << depth_camera_calibration_filename << std::endl;
    return;
  }

  double tmp;
  in >> m_fx_depth;//(0,0)
  in >> tmp; //(0,1);
  in >> m_cx_depth;//(0,2);
  in >> tmp; //(1,0);
  in >> m_fy_depth;//(1,1);
  in >> m_cy_depth;//(1,2);
  m_fx_depth_inv = 1.0f / m_fx_depth;
  m_fy_depth_inv = 1.0f / m_fy_depth;

  in >> tmp;
  in >> tmp;
  in >> tmp; // skip the three.

  in >> m_depth_to_rgb_transform(0, 0);
  in >> m_depth_to_rgb_transform(0, 1);
  in >> m_depth_to_rgb_transform(0, 2);
  in >> m_depth_to_rgb_transform(0, 3);
  in >> m_depth_to_rgb_transform(1, 0);
  in >> m_depth_to_rgb_transform(1, 1);
  in >> m_depth_to_rgb_transform(1, 2);
  in >> m_depth_to_rgb_transform(1, 3);
  in >> m_depth_to_rgb_transform(2, 0);
  in >> m_depth_to_rgb_transform(2, 1);
  in >> m_depth_to_rgb_transform(2, 2);
  in >> m_depth_to_rgb_transform(2, 3);
  m_depth_to_rgb_transform(3, 3) = 1.0;
  in.close();
}

void DataImage::Initialize(Utils::Configuration configuration) {
  m_is_loaded_from_sequence = configuration.read<bool> ("loaded_from_sequence", false);
  m_sequence_width = configuration.read<int> ("image_width", 0);
  m_sequence_height = configuration.read<int> ("image_height", 0);
  m_num_gradient_bins = configuration.read<int> ("gradient_angle_bins", 0);
  m_alternative_gradient_distribution_type = configuration.read<bool> ("alternative_gradient_distribution_type", false);

  //If specified, an alternative gauss mask is used to distribute the votes in the gradient bins.
  if(m_alternative_gradient_distribution_type) {
    float gauss_mask_alt[5][5] = {  {0.0000,    0.0000,    0.0002,    0.0000,    0.0000}  ,
      {0.0000,    0.0113,    0.0837,    0.0113,    0.0000}  ,
      {0.0002,    0.0837,    0.6187,    0.0837,    0.0002}  ,
      {0.0000,    0.0113,    0.0837,    0.0113,    0.0000}  ,
      {0.0000,    0.0000,    0.0002,    0.0000,    0.0000}
    };

    for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < 5; ++j) {
        gauss_mask[i][j] = gauss_mask_alt[i][j];
      }
    }
  }

  m_num_normal_neighbors = configuration.read<int> ("normal_NN_count", 0);
  m_use_rectification_instead_accelerometer = configuration.read<bool> ("no_accel_but_rectification", false);

  //Set intrinsic params!
  std::string rgb_calib_filename = configuration.read<std::string> ("color_calibration_filename", "");
  std::string depth_calib_filename = configuration.read<std::string> ("depth_calibration_filename", "");
  if(rgb_calib_filename==std::string("") || depth_calib_filename==std::string("")){
    //No calibration filenames are set.
    m_has_calibration=false;
  }else{
    SetRGBAndDepthIntrinsic(rgb_calib_filename, depth_calib_filename);
    m_has_calibration=true;
  }

  // Read the configuration paramaters
  m_baseline_focal = configuration.read<float> ("baseline_focal", 0.0f);
  m_kinect_disparity_offset = configuration.read<float> ("kinect_disparity_offset", 0.0f);
  m_kinect_basic_covariance << configuration.read<float> ("kinect_sigma_u", 0.0f) , 0 , 0 ,
                            0 , configuration.read<float> ("kinect_sigma_v", 0.0f) , 0 ,
                            0 , 0 , configuration.read<float> ("kinect_sigma_d", 0.0f) ;


  //Load the directories containing the segmented images.
  //  int s=0;
  //  while(true){
  //    //Create the name of the key
  //    std::stringstream key;
  //    key << "segment_directory" << s;

  //    //Try to load it.
  //    std::string value = configuration.read<std::string>(key.str(), std::string(""));
  //    if(value == ""){
  //      break;
  //    }else{
  //      m_segmentation_directories.push_back(value);
  //    }
  //    s++;
  //  }
}


void DataImage::GetSequenceRgbParameters(double& fx, double& fy, double& cx, double& cy, int& width, int& height) {
  if(m_is_loaded_from_sequence) {
    fx = m_fx_rgb;
    fy = m_fy_rgb;
    cx = m_cx_rgb;
    cy = m_cy_rgb;
    width = m_sequence_width;
    height = m_sequence_height;
  } else {
    throw std::runtime_error("ERROR: Trying to use a single image as if it was loaded from a sequence!");
  }
}


cv::Mat DataImage::ProcessRawDepthImage(cv::Mat raw_depth) {
  if(!m_has_calibration){
    std::runtime_error("Error: Trying to use a calibration while it is not set.");
  }
  
  const int width = raw_depth.cols;
  const int height = raw_depth.rows;
  // Reproject the depth image onto the rgb plane and save the values as float values.
  cv::Mat depth_map(height, width, CV_32FC1, std::numeric_limits<float>::quiet_NaN());//Everything is nan at first
  double x3, y3, z3, tempx3, tempy3, tempz3 = 0;
  int x_proj, y_proj = 0;

  for(int y = 0; y < height; y++) {
    unsigned short* raw_depth_value = raw_depth.ptr<unsigned short> (y);

    for(int x = 0; x < width; x++, raw_depth_value++) {
      //swap the bytes of the depth values.
      unsigned short depth = *raw_depth_value;
      depth = (depth & 0b0000000011111111) << 8 | (depth & 0b11111111100000000) >> 8;

      if(depth != 0b0000011111111111) {       //NaN depth hole in the image, do nothing
        z3 =  m_baseline_focal / (m_kinect_disparity_offset - *raw_depth_value);

        if(z3 > 10 || z3 <= 0) {
          continue;    //Limit depth to 10m. Anything over 10m is unreliable data anyways.
        }

        x3 = (x - m_cx_depth) * z3 * m_fx_depth_inv;
        y3 = (y - m_cy_depth) * z3 * m_fy_depth_inv;

        //transform and backproject.
        tempx3 = x3 * m_depth_to_rgb_transform(0, 0) + y3 * m_depth_to_rgb_transform(0, 1) + z3 * m_depth_to_rgb_transform(0, 2) + m_depth_to_rgb_transform(0, 3);
        tempy3 = x3 * m_depth_to_rgb_transform(1, 0) + y3 * m_depth_to_rgb_transform(1, 1) + z3 * m_depth_to_rgb_transform(1, 2) + m_depth_to_rgb_transform(1, 3);
        tempz3 = x3 * m_depth_to_rgb_transform(2, 0) + y3 * m_depth_to_rgb_transform(2, 1) + z3 * m_depth_to_rgb_transform(2, 2) + m_depth_to_rgb_transform(2, 3);

        x_proj = (tempx3 * m_fx_rgb) / tempz3 + m_cx_rgb + 0.5f;
        y_proj = (tempy3 * m_fy_rgb) / tempz3 + m_cy_rgb + 0.5f;   //added 0.5 for rounding.

        //Check if the point is within the image plain
        if((x_proj >= 0) & (x_proj < width) & (y_proj >= 0) & (y_proj < height)) {
          if(isnan(depth_map.at<float>(y_proj, x_proj)) ||  depth_map.at<float>(y_proj, x_proj) > tempz3) {
            depth_map.at<float>(y_proj, x_proj, 0) = tempz3;    //The old point is covered by the new one.
          }
        }
      }
    }
  }

  return depth_map;
}

Eigen::Matrix3f DataImage::LoadAccelerometerRotation(std::string filename) {
  if(m_use_rectification_instead_accelerometer) {    //Load the precomputed rectification
    //Not nice, but will work for now, load the rectification transform instead of using the accelerometer data.
    filename.replace(filename.find("accelerometer"), 13, std::string("rectification"));

    std::ifstream input_file_stream(filename.c_str());
    Eigen::Matrix3f transformation;

    if(input_file_stream.is_open()) {
      for(int32_t i = 0; i < 3; i++) {
        std::string line = "";
        getline(input_file_stream, line);
        double a = 0;
        double b = 0;
        double c = 0;
        sscanf(line.c_str(), "%lf, %lf, %lf \n", &a, &b, &c);
        transformation(i, 0) = a;
        transformation(i, 1) = b;
        transformation(i, 2) = c;
      }
    } else {
      std::cerr << "Rectification file could not be found. Filename: " << filename << std::endl;
      throw std::runtime_error("");
    }

    input_file_stream.close();
    return transformation;

  } else { // Use the accelerometer data.
    std::vector<float> acl(4, 0);

    if(filename.substr(filename.find_last_of(".") + 1) == "txt") {
      //Load the cloud from the file
      std::ifstream source;                    // build a read-Stream
      source.open(filename.c_str(), std::ios_base::in);    // open data
      float d;
      int i = 0;

      if(!source)  {                      // if it does not work
        std::cerr << "Accelerometer file could not be found. Filename: " << filename << std::endl;
        exit(EXIT_FAILURE);
      } else {
        std::string line;

        while(getline(source, line)) {       //read stream line by line
          if(EOF != sscanf(line.data(), "%f", &d)) {
            acl[i] = d;
            i++;
          }
        }

        source.close();
      }
    } else if(filename.substr(filename.find_last_of(".") + 1) == "dump") {
      freenect_raw_tilt_state state;
      FILE* fp = fopen(filename.c_str(), "r");

      if(fp == NULL) {
        std::cerr << "Accelerometer file could not be found. Filename: " << filename << std::endl;
        exit(EXIT_FAILURE);
      }

      size_t ignore = fread(&state, sizeof(state), 1, fp);
      acl[0] = state.accelerometer_x;
      acl[1] = state.accelerometer_y;
      acl[2] = state.accelerometer_z;
      acl[3] = state.tilt_angle;

      fclose(fp);

    } else {
      std::cerr << "Error Loading the Accelerometer file. Only .txt and .dump files are supported!";
      exit(EXIT_FAILURE);
    }

    // std::cout << " roll, yaw, pitch, kinect pitch angle: "<< acl[0] << " , " << acl[1] << " , " << acl[2] << " , "<< acl[3] << std::endl;

    float pitch_angle = -sign(acl[2]) * acos(acl[1] / (sqrt(pow(acl[1], 2) + pow(acl[2], 2))));
    pitch_angle += (acl[3] * DEGREES2RADIANS);
    //   std::cout << "pitch angle " << pitch_angle << std::endl;
    float roll_angle = -sign(acl[0]) * acos(acl[1] / (sqrt(pow(acl[1], 2) + pow(acl[0], 2))));
    //  std::cout << "roll angle " << roll_angle << std::endl;

    Eigen::Matrix3f R_pitch;
    R_pitch.setZero(3, 3);
    R_pitch(0, 0) = 1;
    R_pitch(1, 1) = cos(pitch_angle);
    R_pitch(2, 2) = R_pitch(1, 1);
    R_pitch(2, 1) = sin(pitch_angle);
    R_pitch(1, 2) = -R_pitch(2, 1);
    //    std::cout << "\npitch rotation: \n "<< R_pitch << std::endl;

    Eigen::Matrix3f R_roll;
    R_roll.setZero(3, 3);
    R_roll(2, 2) = 1;
    R_roll(0, 0) = cos(roll_angle);
    R_roll(1, 1) = R_roll(0, 0);
    R_roll(1, 0) = sin(roll_angle);
    R_roll(0, 1) = -R_roll(1, 0);
    //   std::cout << "\nroll rotation: \n"<< R_roll << std::endl;

    Eigen::Matrix3f R(3, 3);
    R = R_roll *  R_pitch;
    //   std::cout << "\n final rotation: \n"<< R << std::endl;
    return  R;
  }

}

void DataImage::ComputeDepthNormalization() {
  /*Eigen::Matrix3f  intrinsic_camera_matrix;
  intrinsic_camera_matrix <<  m_fx_rgb, 0,        m_cx_rgb,
                              0,        m_fy_rgb, m_cy_rgb,
                              0,        0,        1;

  Eigen::Matrix3f homography = intrinsic_camera_matrix * m_accelerometer_rotation * intrinsic_camera_matrix.inverse();
  //Find new min and max bounds.
  float min = 1;
  float max = m_width;

  Eigen::Vector3f point;
  Eigen::Vector3f projected;

  point << 1 , 1 , 1;
  projected = homography * point;
  projected = projected / projected(2);
  if(projected(0) > max) max = projected(0);
  if(projected(0) < min) min = projected(0);


  point << m_width , 1 , 1;
  projected = homography * point;
  projected = projected / projected(2);
  if(projected(0) > max) max = projected(0);
  if(projected(0) < min) min = projected(0);


  point << 1 , m_height , 1;
  projected = homography * point;
  projected = projected / projected(2);
  if(projected(0) > max) max = projected(0);
  if(projected(0) < min) min = projected(0);


  point << m_width , m_height , 1;
  projected = homography * point;
  projected = projected / projected(2);
  if(projected(0) > max) max = projected(0);
  if(projected(0) < min) min = projected(0);

  std::vector<float> max_counter(1+  (int)(-round(min)) + 1 + (int)(round(max))+1, 0); //1 at left side for safety, 1 for zero and 1 at right side for safty
  const int offset = -round(min)+2; //+2 in order to center it correctly.
  Image<int> max_bin(m_width, m_height, 1);
  int index=0;
  for(int y=1; y<= m_height; y++){
    for(int x=1; x<= m_width; x++){
      point << x , y, 1;
      projected = homography * point;
      projected /= projected(2);
      index = offset + round(projected(0));
      max_bin(x-1,y-1) = index;
      if(max_counter[index]< m_depth_image(x-1,y-1))
        max_counter[index]= m_depth_image(x-1,y-1);
    }
  }

  for(int i=0; i<max_counter.size(); i++){
    max_counter[i]=1/max_counter[i];
  }
  Image<float> temp(m_width, m_height, 1);
  for(int y=0; y< m_height; y++){
    for(int x=0; x< m_width; x++){
      temp(x,y) = m_depth_image(x,y) * max_counter[max_bin(x,y)];
    }
  }
  m_normalized_depth_image = temp;
  */

  //Find the y-coordinate region where the deepest points are.
  /*  std::vector<int> y_coordinates(480,0);

  std::vector<float> temp(640,0);

  for(int x=0; x< 640; ++x){
    int max_y=-1;
    for(int y=0; y < 480; ++y){
      if(temp[x] < m_depth_image(x,y))
        temp[x] = m_depth_image(x,y);
        max_y = y;
    }
    y_coordinates[max_y]++;
  }

  //Go over the array in a windowed mannor.

  int max_y = -1;
  int max_y_value = 0;
  int window_size = 10;
  int value=0;
  for(int y = 0 ; y < m_height; ++y){
    if(y < window_size){
      value += y_coordinates[y];

    }else{
      value += y_coordinates[y];
      value -= y_coordinates[y-window_size];
    }

    if(value > max_y_value){
      max_y_value = value;
      max_y = y;
    }
  }

  //We have a maximum y, look for the maximum around this scanline.
  int min = max_y - 2*window_size;
  min = min >0 ? min : 0;
  int max = max_y + 2*window_size;
  max = max  < m_height ? max : m_height -1; */
  std::vector<float> temp2(m_width, 0);

  for(int y = 0; y < m_height; ++y) {
    for(int x = 0; x < m_width; ++x) {
      float depth = m_depth_image.at<float> (y, x); 
      Eigen::Vector3f point;
      point << (static_cast<float>(x) - m_cx_rgb) * m_fx_rgb_inv * depth,
        (static_cast<float>(y) - m_cy_rgb) * m_fy_rgb_inv * depth,
        depth;
      Eigen::Vector3f rectified = m_accelerometer_rotation * point;
      if(temp2[x] < rectified(2)){
        temp2[x] = rectified(2);
      }
    }
  }

  for(int x = 0; x < m_width; ++x) {
    temp2[x] = 1 / temp2[x];
  }

  m_normalization_depth_factor =  temp2;
}

void DataImage::Calculate3DFeature() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

  //Get a point cloud.
  if(m_has_point_cloud){
    cloud = m_cloud;
  }else if(m_has_depth){
    cloud = ComputePointCloudFromDepth();
  }else{
    throw std::runtime_error("Error: Could not compute the geometric features as no depth image was loaded!");
  }


  std::vector<int32_t> indices;
  indices.resize(cloud->points.size());

  for(size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }

  boost::shared_ptr<std::vector<int32_t> > indicesptr(new std::vector<int32_t> (indices));

  //pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  pcl::search::KdTree<pcl::PointXYZ> kdtree;

  kdtree.setInputCloud(cloud);

  cv::Mat temp_3d_features(m_height, m_width, CV_32FC3);
  // Image<double> normals(m_width, m_height, 3);
  //temp_3d_features.Fill(0.0f);
  int index = 0;

  for(int y = 0; y < m_height; y++) {
    float* p = temp_3d_features.ptr<float> (y);
    bool* valid = m_validity_mask.ptr<bool> (y);

    for(int x = 0; x < m_width; x++, valid++) {
      if(*valid) {
        {
          std::vector<int> nn_indices(m_num_normal_neighbors);
          std::vector<float> nn_dists(m_num_normal_neighbors);

          if(kdtree.nearestKSearch(cloud->at(index), m_num_normal_neighbors, nn_indices, nn_dists) == 0) {
            std::cout << "normal estimator failed..." << std::endl;
            continue;
          }

          Eigen::Vector4f centroid;
          Eigen::Matrix3f covariance_matrix;
          pcl::compute3DCentroid(*cloud, nn_indices, centroid);
          pcl::computeCovarianceMatrix(*cloud, nn_indices, centroid, covariance_matrix);
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covariance_matrix);

          float sigma_p = es.eigenvalues()(0);
          float sigma_l = (es.eigenvalues()(2) - es.eigenvalues()(1));         ///es.eigenvalues()(2);
          float sigma_s = (es.eigenvalues()(1) - es.eigenvalues()(0));         ///es.eigenvalues()(2);
          //Eigen::Vector3f vn = es.eigenvectors().col(0);
          //Eigen::Vector3f vt = es.eigenvectors().col(2);
          // normal_small = vt;
          //          temp_3d_features(x,y,0) = sigma_p;
          //          temp_3d_features(x,y,1) = sigma_s;
          //          temp_3d_features(x,y,2) = sigma_l;
          p[x * 3] = sigma_p;
          p[x * 3 + 1] = sigma_s;
          p[x * 3 + 2] = sigma_l;

          // temp(x,y,4) = (vn(1) / vn.norm()) * (sigma_l / std::max(sigma_l, std::max(sigma_p, sigma_s)));
          //temp(x,y,5) = (vt(1) / vt.norm()) * (sigma_s / std::max(sigma_l, std::max(sigma_p, sigma_s)));
        }
        index++;
      }
    }
  }

  m_3d_feature = temp_3d_features;
  m_has_3d_features = true;
}


void DataImage::Compute3DCovariance() {
  if(!m_has_depth) {
    throw std::runtime_error("Error: Could not compute 3D covariance as no depth image was loaded!");
  }

  if(!m_has_calibration){
    std::runtime_error("Error: Trying to use a calibration while it is not set.");
  }

  m_3d_covariance = cv::Mat(m_height, m_width, CV_32FC(9));

  Eigen::Matrix3f covariance;
  Eigen::Matrix3f jacobian;
  bool* valid;
  float* cov_pointer;

  for(int y = 0; y < m_height; y++) {
    valid = m_validity_mask.ptr<bool> (y);
    cov_pointer = m_3d_covariance.ptr<float> (y);

    for(int x = 0; x < m_width; x++, valid++, cov_pointer += 9) {
      if(*valid) {
        float depth =  m_depth_image.at<float> (y, x);
        //TODO Check this! Crazy things down here O.o;
        float depth_inv = 1.0f / (m_baseline_focal * depth);
        float u_relative = static_cast<float>(x) - m_cx_rgb;
        float v_relative = static_cast<float>(y) - m_cy_rgb;

        jacobian << depth* m_fx_rgb_inv , 0                , u_relative* m_fx_rgb_inv* depth_inv ,
                 0                 , depth* m_fy_rgb_inv , v_relative* m_fy_rgb_inv* depth_inv ,
                 0                 , 0                , depth_inv;

        covariance = jacobian * m_kinect_basic_covariance * jacobian.transpose();

        cov_pointer[0] = covariance(0);
        cov_pointer[1] = covariance(1);
        cov_pointer[2] = covariance(2);
        cov_pointer[3] = covariance(3);
        cov_pointer[4] = covariance(4);
        cov_pointer[5] = covariance(5);
        cov_pointer[6] = covariance(6);
        cov_pointer[7] = covariance(7);
        cov_pointer[8] = covariance(8);

      }
    }
  }

  m_has_3d_covariance = true;
}

void DataImage::ComputeNormals() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

  //Get a point cloud. 
  if(m_has_point_cloud){
    cloud = m_cloud;
  }else if(m_has_depth){
    cloud = ComputePointCloudFromDepth();
  }else{
    throw std::runtime_error("Error: Could not compute normal vectors as no depth image or point cloud was loaded!");
  }
  
  // estimate normals
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

  pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
  ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);    //From ishrats code, is it good ?
  //ne.setRectSize ( 17, 17 ); //Does nothing??
  //ne.setMaxDepthChangeFactor(0.015f);
  ne.setMaxDepthChangeFactor(0.04f);
  ne.setNormalSmoothingSize(10.0f);
  ne.setInputCloud(cloud);
  ne.compute(*normals);


  cv::Mat temp(m_height, m_width, CV_32FC3);

  for(int y = 0; y < m_height; y++) {
    float* normal_ptr = temp.ptr<float> (y);

    for(int x = 0; x < m_width; x++, normal_ptr += 3) {
      normal_ptr[0] = normals->at(x, y).normal_x;
      normal_ptr[1] = normals->at(x, y).normal_y;
      normal_ptr[2] = normals->at(x, y).normal_z;
    }
  }

  m_normals = temp;
  m_has_normals = true;
}


std::string DataImage::GetUnaryFilename() const {
  return m_unary_filename;
}
void DataImage::SetUnaryFilename(const std::string& unary_filename) {
  m_unary_filename = unary_filename;
}

std::string DataImage::GetResultFilename() const {
  return m_result_filename;
}

void DataImage::SetResultFilename(const std::string& result_filename) {
  m_result_filename = result_filename;
}

std::string DataImage::GetPointCloudFilename() const {
  return m_pcd_filename;
}

void DataImage::SetPointCloudFilename(const std::string& pcd_filename) {
  m_pcd_filename = pcd_filename;
}

void DataImage::AddPointCloud(pcl::PointCloud< pcl::PointXYZ >::Ptr cloud) {
  m_cloud = cloud;
  m_has_point_cloud = true;
}

pcl::PointCloud< pcl::PointXYZ >::Ptr DataImage::GetPointCloud() const {
  if(m_has_point_cloud){
    return m_cloud;
  }else{
    throw std::runtime_error("A pointcloud was required, but it was not loaded for the image. ");
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr DataImage::ComputePointCloudFromDepth() {
  // load point cloud
  
  if(!m_has_calibration){
    std::runtime_error("Error: Trying to use a calibration while it is not set.");
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->height = m_height;
  cloud->width = m_width;
  cloud->points.resize(m_width * m_height);
  cloud->is_dense = false;

  for(int y = 0; y < m_height; y++) {
    for(int x = 0; x < m_width; x++) {
      pcl::PointXYZ point;
      point.z = m_depth_image.at<float> (y, x);
      point.x = (static_cast<float>(x) - m_cx_rgb) * m_fx_rgb_inv *  point.z;
      point.y = (static_cast<float>(y) - m_cy_rgb) * m_fy_rgb_inv *  point.z;
      cloud->at(x, y) = point;
    }
  }
  return cloud;
}


