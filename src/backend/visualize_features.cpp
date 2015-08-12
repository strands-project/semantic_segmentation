// Local includes
#include "utils/calibration.h"
#include "utils/config.h"
#include "utils/commandline_parser.h"
#include "utils/cv_util.h"
#include "utils/rgb_label_conversion.h"
#include "utils/data_loader.h"
#include "voxel.h"

// Eigen includes
#include <Eigen/Core>

// libforest
#include "libforest/libforest.h"

// OpenCV includes
#include <opencv2/opencv.hpp>

// PCL includes
#include <pcl/common/time.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

// STL includes
#include <map>
#include <memory>



std::string usage(char * exe){
  std::string u("\n");
  u += std::string(exe) + " --conf <config file> --opt1 <val1> ... \n";
  return u;
}

int main (int argc, char ** argv) {
  if(argc <= 2){
    throw std::runtime_error("No parameters given. Usage: " + usage(argv[0]));
  }

  //Parse all parameters
  std::map<std::string, std::string> parameter_name_value_map;
  bool parse_succeeded = Utils::parseParamters(argc, argv, parameter_name_value_map);
  if(!parse_succeeded){
    throw std::runtime_error("Mangled command line arguments. " + usage(argv[0]));
  }

  //check if we found a config file.
  if(parameter_name_value_map.count("conf") == 0){
    throw std::runtime_error("No config file was given" + usage(argv[0]));
  }

  std::string config_file = parameter_name_value_map["conf"];
  parameter_name_value_map.erase("conf");
  Utils::Config conf(config_file, parameter_name_value_map);

  Utils::DataLoader dl(conf);

  Utils::RgbLabelConversion label_converter = dl.getLabelConverter();

  std::vector<std::string> image_names = dl.getImageList(conf.get<std::string>("key","test_images"));

  //For each image
  for(std::string image_name : image_names){
    //Load the image
    cv::Mat color = dl.loadColor(image_name);
    cv::Mat depth = dl.loadDepth(image_name);
    cv::Mat label = dl.loadLabel(image_name);
    Utils::Calibration calib = dl.loadCalibration(image_name);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_unrectified;
    dl.create_cloud(depth, color, calib, cloud, cloud_unrectified);

    //Get the voxels, we don't use the stored ones to easily visualize the change of parameters.
    std::map<int, std::shared_ptr<Voxel> > voxels;
    dl.extractVoxels(cloud, cloud_unrectified, voxels);

    int n = voxels[1]->getFeatures().rows(); //Just to get the size once.

    cv::Mat result_image(cloud->height, cloud->width, CV_32FC1, cv::Scalar(0));
    for(int j = 0; j < n; j++){
      for(auto v : voxels){
        libf::DataPoint f = v.second->getFeatures();
        v.second->drawValueIntoImage<float>(result_image, f(j));
      }
      Utils::ShowCvMatHeatMap(result_image);
    }
    cv::Mat label_image(cloud->height, cloud->width, CV_32SC1, cv::Scalar(-1));
    for(auto v : voxels){
      v.second->computeLabel(label);
      v.second->drawLabelIDIntoImage(label_image);
    }
    Utils::ShowCvMat(label_converter.labelToRgb(label_image));
  }
  return 0;
}
