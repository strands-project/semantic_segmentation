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
  std::vector<std::string>  image_sets;
  image_sets.push_back("test_images");
  image_sets.push_back("train_images");

  for(std::string  set : image_sets){
    //Get the image list.
    std::vector<std::string> image_list = dl.getImageList(set);

    //for each image
    for(std::string file : image_list){

      //load the image data
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_unrectified;
      dl.create_cloud(file, cloud, cloud_unrectified);

      //voxelize
      std::map<int, std::shared_ptr<Voxel> > voxels;
      dl.extractVoxels(cloud, cloud_unrectified, voxels);

      //Save the image to disk.
      //Project
      cv::Mat result_image(cloud->height, cloud->width, CV_32SC1, cv::Scalar(0));
      for(auto v : voxels){
        v.second->drawValueIntoImage<int>(result_image, v.second->getVoxelID());
      }

      //Save
      cv::imwrite(dl.getVoxelName(file), Utils::segmentIdToBgr(result_image));
    }
  }
  return 0;
}
