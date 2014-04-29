// C includes
#include <dirent.h>

// STL includes
#include <stdexcept>

// Boost includes
#include <boost/progress.hpp>

// Local includes
#include "../configuration.hh"
#include "strands.h"
#include "../color_coding/general_color_coding.h"
#include "../string_util.h"
#include "../cv_util.hh"

using namespace Utils;

StrandsDataset::StrandsDataset(Configuration conf) : StaticDataset::StaticDataset(conf), DynamicDataset::DynamicDataset(conf) {
  m_color_coding = new GeneralColorCoding();
  m_configuration = conf;

}

StrandsDataset::~StrandsDataset() {
  delete m_color_coding;
}

void StrandsDataset::Load(DataType data_type, int load_flags) {
  //Check if we are reading images from the simple dataset, or if we are reading images
  //from a sequence.
  m_load_requirement_flags = load_flags;

  // TODO init the dataimage class!
  Utils::DataImage::Initialize(m_configuration);

    //Load all filenames from a file specified in the configuration.
    m_image_filenames = List(data_type);
    m_image_count = m_image_filenames.size();
    // Load all needed folder and extensions.
    m_images_extension = m_configuration.read<std::string>("image_extension");
    m_images_folder = m_configuration.read<std::string>("image_directory");

    m_annotations_extension = m_configuration.read<std::string>("annotation_extension");
    m_annotations_folder = m_configuration.read<std::string>("annotation_directory");

    m_depth_images_extension = m_configuration.read<std::string>("depth_extension");
    m_depth_images_folder = m_configuration.read<std::string>("depth_directory");
    m_accelerometer_extension = m_configuration.read<std::string>("accelerometer_extension", "");
    m_accelerometer_folder = m_configuration.read<std::string>("accelerometer_directory", "");

    m_unary_extension = m_configuration.read<std::string>("unary_extension");
    m_unary_folder = m_configuration.read<std::string>("unary_directory");


    m_result_extension = m_configuration.read<std::string>("result_extension");
    m_result_folder = m_configuration.read<std::string>("result_directory");

    m_pcd_extension = m_configuration.read<std::string>("pcd_extension");
    m_pcd_folder = m_configuration.read<std::string>("pcd_directory");



  m_color_calib_filename = m_configuration.read<std::string>("color_calibration_filename");
  m_depth_calib_filename = m_configuration.read<std::string>("depth_calibration_filename");

}

void StrandsDataset::Load(){
  //Nothing to load yet for the dynamic case.
  Utils::DataImage::Initialize(m_configuration);
}


Utils::DataImage StrandsDataset::GenerateImage(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, const Eigen::Matrix3f& q) {
  Utils::DataImage image;

  //Set the cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_xyz_ptr->width = cloud.width;
  cloud_xyz_ptr->height = cloud.height;
  cloud_xyz_ptr->is_dense = cloud.is_dense;
  cloud_xyz_ptr->resize(cloud.width * cloud.height);
  for(size_t i = 0; i < cloud.points.size(); i++) {
    cloud_xyz_ptr->points[i].x = cloud.points[i].x;
    cloud_xyz_ptr->points[i].y = cloud.points[i].y;
    cloud_xyz_ptr->points[i].z = cloud.points[i].z;
  }

  image.AddPointCloud(cloud_xyz_ptr);

  //Extract RGB and set it.
  if(m_load_requirement_flags & Utils::LAB) {
    //First add the rgb image, this sets the width and the height.
    int index = 0;
    cv::Mat bgr = cv::Mat(cloud.height, cloud.width, CV_8UC3);

    for(int y = 0; y < cloud.height; ++y) {
      unsigned char* color_ptr = bgr.ptr<unsigned char>(y);

      for(int x = 0; x < cloud.width; ++x) {
        *color_ptr++ = cloud.points[index].b;
        *color_ptr++ = cloud.points[index].g;
        *color_ptr++ = cloud.points[index].r;
        index++;
      }
    }

    image.AddRGBImage(bgr);

    //Compute the LAB image, but only keep the RGB image if it is needed.
    image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);

    if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
      //Also compute an integral image!
      image.ComputeColorIntegralImage();
    }

  } else {
    //Just set the RGB image if it is needed.
    if(m_load_requirement_flags & Utils::RGB) {
      int index = 0;
      cv::Mat bgr = cv::Mat(cloud.height, cloud.width, CV_8UC3);

      for(int y = 0; y < cloud.height; ++y) {
        unsigned char* color_ptr = bgr.ptr<unsigned char>(y);

        for(int x = 0; x < cloud.width; ++x) {
          *color_ptr++ = cloud.points[index].b;
          *color_ptr++ = cloud.points[index].g;
          *color_ptr++ = cloud.points[index].r;
          index++;
        }
      }
      image.AddRGBImage(bgr);
    }
  }

  //Extract Depth and set it.
  if(m_load_requirement_flags & Utils::DEPTH) {
    //Load the depth image.
    int index = 0;
    cv::Mat depth = cv::Mat(cloud.height, cloud.width, CV_32FC1, std::numeric_limits<float>::quiet_NaN());

    for(int y = 0; y < cloud.height; ++y) {
      float* depth_ptr = depth.ptr<float>(y);
      for(int x = 0; x < cloud.width; ++x) {
        *depth_ptr++ = cloud.points[index].z;
        index++;
      }
    }

    image.AddDepthImage(depth);
    if(m_load_requirement_flags & Utils::ACCELEROMETER) {
      //Load the accelerometer data.
      image.AddAccelerometerData(q);
    }

    if(m_load_requirement_flags & Utils::DEPTH_COVARIANCE) {
      throw std::runtime_error("Cannot compute the depth covariance for data loaded from point clouds, the calibration is needed!");
    }

  }

  //Do everything else.
  if(m_load_requirement_flags & Utils::UNARY || m_load_requirement_flags & Utils::UNARY2 ||
     m_load_requirement_flags & Utils::ANNOTATION) {
    throw std::runtime_error("Images loaded from a point cloud do not have unary potentials or annotations!");
  }


  if(m_load_requirement_flags & Utils::GEOMETRIC_FEAT) {
    image.Calculate3DFeature();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_COLOR) {
    image.ComputeLabGradientImages();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_DEPTH) {
    image.ComputeDepthGradientImages();
  }

  if(m_load_requirement_flags & Utils::NORMALS) {
    image.ComputeNormals();
  }
  
  return image;
}


Utils::DataImage StrandsDataset::GenerateImage(int index) const {
  //Create an empty image.
  Utils::DataImage image;

    //Load color data.
    if(m_load_requirement_flags & Utils::LAB) {
      //First add the rgb image, this sets the width and the height.
      std::string rgb_fn = m_images_folder + "/" + m_image_filenames[index];
      image.AddRGBImage(rgb_fn);

      //Compute the LAB image, but only keep the RGB image if it is needed.
      image.ComputeLABImage(m_load_requirement_flags & Utils::RGB);

      if(m_load_requirement_flags & Utils::LAB_INTEGRAL) {
        //Also compute an integral image!
        image.ComputeColorIntegralImage();
      }

    } else {
      //Just set the RGB image if it is needed.
      if(m_load_requirement_flags & Utils::RGB) {
        std::string rgb_fn = m_images_folder + "/" + m_image_filenames[index];
        image.AddRGBImage(rgb_fn);
      }
    }

    //Load depth data.
    if(m_load_requirement_flags & Utils::DEPTH) {
      //Load the depth image.
      std::string depth_fn = m_depth_images_folder + "/" +  m_image_filenames[index];
      SwapExtension(depth_fn, m_images_extension, m_depth_images_extension);
      image.AddDepthImage(depth_fn);

      if(m_load_requirement_flags & Utils::ACCELEROMETER) {
        //Load the accelerometer data.
        std::string accel_fn = m_accelerometer_folder + "/" +  m_image_filenames[index];
        SwapExtension(accel_fn, m_images_extension, m_accelerometer_extension);
        image.AddAccelerometerData(accel_fn);
      }
    }

    //Load unary potentials and set the filename.
    std::string unary1_fn = m_unary_folder + "/" + m_image_filenames[index];
    SwapExtension(unary1_fn, m_images_extension, m_unary_extension);
    image.SetUnaryFilename(unary1_fn);

    if(m_load_requirement_flags & Utils::UNARY) {
      image.AddUnaryPotential(unary1_fn);
    }

    //Load the ground truth annotation.
    std::string annotation_fn = m_annotations_folder + "/" + m_image_filenames[index];
    SwapExtension(annotation_fn, m_images_extension, m_annotations_extension);
    image.SetAnnotationFilename(annotation_fn);

    if(m_load_requirement_flags & Utils::ANNOTATION) {
      image.AddAnnotation(annotation_fn, m_color_coding);
    }

    //Set the result filename.
    std::string result_fn = m_result_folder + "/" + m_image_filenames[index];
    SwapExtension(result_fn, m_images_extension, m_result_extension);
    image.SetResultFilename(result_fn);

    //Set the pcd filename.
    std::string pcd_fn = m_pcd_folder + "/" + m_image_filenames[index];
    SwapExtension(pcd_fn, m_images_extension, m_pcd_extension);
    image.SetPointCloudFilename(pcd_fn);



  if(m_load_requirement_flags & Utils::GEOMETRIC_FEAT) {
    image.Calculate3DFeature();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_COLOR) {
    image.ComputeLabGradientImages();
  }

  if(m_load_requirement_flags & Utils::GRADIENT_DEPTH) {
    image.ComputeDepthGradientImages();
  }

  if(m_load_requirement_flags & Utils::NORMALS) {
    image.ComputeNormals();
  }

  return image;
}

void StrandsDataset::SetLoadFlags(int load_flags){
  m_load_requirement_flags = load_flags;
}

