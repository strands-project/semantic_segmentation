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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>


// STL includes
#include <map>
#include <memory>



std::string usage(char * exe){
  std::string u("\n");
  u += std::string(exe) + " --conf <config file> --opt1 <val1> ... \n";
  return u;
}

void rectify_images(Utils::Config& conf){

  Utils::RgbLabelConversion label_converter(conf.getJsonValueAsString(conf.get<std::string>("color_coding_key")));
  std::vector<std::string> image_names;
  cv::Mat colors;
  cv::Mat depths;
  cv::Mat labels;
  Utils::Calibration calibrations;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr clouds_unrectified;

  image_names = conf.get<std::vector<std::string> >("train_images");
  std::string color_dir = conf.getPath("color_dir");
  std::string color_ext = conf.get<std::string>("color_ext");
  std::string depth_dir = conf.getPath("depth_dir");
  std::string depth_ext = conf.get<std::string>("depth_ext");
  std::string label_dir = conf.getPath("label_dir");
  std::string label_ext = conf.get<std::string>("label_ext");
  std::string calibration_dir = conf.getPath("calibration_dir");
  std::string calibration_ext = conf.get<std::string>("calibration_ext");

  for(std::string im_name : image_names){
    depths =cv::imread(depth_dir + im_name + depth_ext, CV_LOAD_IMAGE_ANYDEPTH);
    colors = cv::Mat(depths.rows, depths.cols, CV_8UC3);
    cv::cvtColor(cv::imread(color_dir + im_name + color_ext), colors, CV_BGR2Lab);
    labels = label_converter.rgbToLabel(cv::imread(label_dir + im_name + label_ext));
    calibrations = Utils::Calibration(calibration_dir + im_name + calibration_ext);
    floor = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    clouds_unrectified = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());


    //Convert to point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld = floor;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld_unr = clouds_unrectified;
    ushort* d_ptr = depths.ptr<ushort>(0);
    cld_unr->height = depths.rows;
    cld_unr->width = depths.cols;
    cld_unr->resize(cld_unr->height*cld_unr->width);
    cld_unr->is_dense = false;
    Eigen::MatrixXf mat(3, cld_unr->size());
    unsigned int index = 0;
    for(unsigned int y = 0; y < cld_unr->height; y++){
      for(unsigned int x = 0; x  < cld_unr->width; x++){
        if(*d_ptr <= 0 || *d_ptr > 10000){
          mat(2, index) = std::numeric_limits<float>::quiet_NaN();
        }else{
          mat(2, index) = *d_ptr  / 1000.0f;
        }
        mat(0, index) = mat(2, index) * x;
        mat(1, index) = mat(2, index) * y;
        index++;
        d_ptr++;
      }
    }

    Eigen::MatrixXf unrect = calibrations._intrinsic_inverse*mat;
    uchar* c_ptr = colors.ptr<uchar>(0);
    int* l_ptr = labels.ptr<int>(0);;
    uint8_t r, g, b;
    uint32_t rgb;
    for(index = 0; index < cld_unr->size(); index++){
      cld_unr->points[index].x = unrect(0,index);
      cld_unr->points[index].y = unrect(1,index);
      cld_unr->points[index].z = unrect(2,index);
      b = c_ptr[0];
      g = c_ptr[1];
      r = c_ptr[2];
      rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
      c_ptr+=3;
      cld_unr->points[index].rgb = *reinterpret_cast<float*>(&rgb);
      if(*l_ptr == 1){
        cld->points.push_back(cld_unr->points[index]);
      }
      l_ptr++;
    }


    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);

    //Give a prior to which plane should be found and limit what deviation is accepted.
    seg.setAxis(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    seg.setDistanceThreshold (0.05);
    seg.setMaxIterations(1000);


    //Fit the plane
    seg.setInputCloud(cld);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    seg.segment (*inliers, *coefficients);


    if (inliers->indices.size () != 0){

      //Get the up direction of the plane.
      Eigen::Vector3f plane_up(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
      float mult = 1;
      if(coefficients->values[1] > 0){
        plane_up *= -1;
        mult = -1;
      }

      //Now we rotate, first we go along one direction, then rectify and do the next.
      Eigen::Affine3f rotation1, rotation2;
      pcl::getTransFromUnitVectorsXY(Eigen::Vector3f(1.0, 0.0, 0.0), plane_up, rotation1);
      pcl::getTransFromUnitVectorsZY(Eigen::Vector3f(0.0, 0.0, 1.0), rotation1*plane_up, rotation2);



      //From this compute the extrinsic data.
      Eigen::Matrix3f extrin = (rotation2*rotation1).matrix().block(0,0,3,3);
      //flip y,z for robotics. Also for some reason we need to invert the y direction then.
      Eigen::Matrix3f flip;
      flip << 1, 0, 0, 0, 0, -1, 0, 1, 0;
      extrin = flip*extrin;
      Eigen::Vector3f translation(0,0,mult*coefficients->values[3]);

      calibrations.setExtrinsics(extrin, translation);
      calibrations.save(calibration_dir.substr(0, calibration_dir.length()-1) +"_rect/" + im_name + calibration_ext);
    }else{
      //For now we'll just skip this file. As no new file is generated, we will just discard this.
    }
  }
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


  rectify_images(conf);

  return 0;
}
