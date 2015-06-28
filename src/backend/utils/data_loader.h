#pragma once

#include "utils/calibration.h"
#include "utils/config.h"
#include "utils/rgb_label_conversion.h"
#include "libforest/libforest.h"

#include "voxel.h"

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace Utils{

class DataLoader {
public:
  DataLoader(Config& conf, bool load_dirs = true);

  ~DataLoader(){};


  void extractVoxelsFromImage(cv::Mat& voxel_image,
                              pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
                              std::map<int, std::shared_ptr<Voxel> >& voxel_storage);

  void extractVoxels(pcl::PointCloud< pcl::PointXYZRGBA>::Ptr cloud,
                     std::map< int, std::shared_ptr< Voxel> >& voxel_storage,
                     pcl::PointCloud< pcl::PointXYZRGBA>::Ptr& voxelized_cloud);

  void extractVoxels(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
                     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_unrectifed,
                     std::map<int, std::shared_ptr<Voxel> >& voxel_storage);

  cv::Mat loadColor(std::string image_name) const;

  cv::Mat loadDepth(std::string image_name) const;

  cv::Mat loadLabel(std::string image_name) const;

  Utils::Calibration loadCalibration(std::string image_name) const;

  cv::Mat loadVoxel(std::string image_name) const;

  void create_cloud(cv::Mat& depth,
                    cv::Mat& color,
                    Utils::Calibration& calibration,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_unrectified);


  void create_cloud(std::string image_name,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud_unrectified);

  std::vector<std::string> getImageList(std::string key);

  std::string getVoxelName(std::string file);

  std::string getResultName(std::string file);


  //Causes errors in libforest when in source file. TODO check this out and post an issue.
  libf::DataStorage::ptr loadAllTrainingData(std::string key, bool use_vccs_rectification = true, bool voxels_from_image = true){
    _image_names = getImageList(key);
    //Create the dataset for DataStorage
    libf::DataStorage::ptr d = libf::DataStorage::Factory::create();

    //Loop over all images
    //     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    for(std::string n : _image_names){
      std::cout << n << std::endl;

      //Load the initial data
      cv::Mat depth = loadDepth(n);
      cv::Mat color = loadColor(n);
      cv::Mat label = loadLabel(n);
      Utils::Calibration calibration = loadCalibration(n);
      cv::Mat voxel;
      if(use_vccs_rectification && voxels_from_image){
        voxel = loadVoxel(n);
      }

      //Create the cloud
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_unrectified;
      create_cloud(depth, color, calibration, cloud, cloud_unrectified);

      //       pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb_render(cloud);
      //       viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, rgb_render, n);
      //       viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, n);
      //       viewer->spin();

      //Voxelize
      std::map<int, std::shared_ptr<Voxel> > current_voxels;
      if(!use_vccs_rectification){ // we'll just recompute
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxelized_cloud;
        extractVoxels(cloud, current_voxels, voxelized_cloud);
      }else if(voxels_from_image){
        extractVoxelsFromImage(voxel,cloud, current_voxels);
      }else{
        extractVoxels(cloud, cloud_unrectified, current_voxels);
      }

      //Extract features
      for(auto v : current_voxels){
        if (v.second->getSize() >= _minimum_point_count) {
          v.second->computeLabel(label);
          int l =  v.second->getLabel();
          if(l >=0){
            v.second->computeFeatures();
            //Store datapoints
            d->addDataPoint(v.second->getFeatures(), l);
          }
        }
      }
    }

    return d;
  }


  Utils::RgbLabelConversion& getLabelConverter();

private:
  Config& _conf;
  float _voxel_resolution;
  float _seed_resolution;
  float _color_importance;
  float _spatial_importance;
  float _normal_importance;
  int _minimum_point_count;

  std::vector<std::string> _image_names;
  std::string _color_dir;
  std::string _color_ext;
  std::string _depth_dir;
  std::string _depth_ext;
  std::string _label_dir;
  std::string _label_ext;
  std::string _calibration_dir;
  std::string _calibration_ext;;
  std::string _voxel_dir;
  std::string _voxel_ext;
  std::string _result_dir;
  std::string _result_ext;

  Utils::RgbLabelConversion _label_converter;
};
}