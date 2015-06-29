#include "data_loader.h"
#include "utils/cv_util.h"

// PCL includes
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>


using namespace Utils;

DataLoader::DataLoader(Utils::Config& conf, bool load_dirs) : _conf(conf) {
  _voxel_resolution = _conf.get<float>("voxel_resolution");
  _seed_resolution = _conf.get<float>("seed_resolution");
  _color_importance = _conf.get<float>("color_importance");
  _spatial_importance = _conf.get<float>("spatial_importance");
  _normal_importance = _conf.get<float>("normal_importance");
  _minimum_point_count = _conf.get<int>("min_point_count");
  if(load_dirs){
    _color_dir = _conf.getPath("color_dir");
    _color_ext = _conf.get<std::string>("color_ext");
    _depth_dir = _conf.getPath("depth_dir");
    _depth_ext = _conf.get<std::string>("depth_ext");
    _label_dir = _conf.getPath("label_dir");
    _label_ext = _conf.get<std::string>("label_ext");
    _voxel_dir = _conf.getPath("voxel_dir");
    _voxel_ext = _conf.get<std::string>("voxel_ext");
    _calibration_dir = _conf.getPath("calibration_dir");
    _calibration_ext = _conf.get<std::string>("calibration_ext");
    _result_dir = _conf.getPath("result_dir");
    _result_ext = _conf.get<std::string>("result_ext");
  }
  _label_converter = Utils::RgbLabelConversion(_conf.getJsonValueAsString(_conf.get<std::string>("color_coding_key")));
}


RgbLabelConversion& DataLoader::getLabelConverter() {
  return _label_converter;
}



std::string DataLoader::getResultName(std::string file) {
  return _result_dir + file + _result_ext;
}


std::string DataLoader::getVoxelName(std::string file) {
  return _voxel_dir + file + _voxel_ext;
}


std::vector< std::string> DataLoader::getImageList(std::string key) {
  return _conf.get<std::vector<std::string> >(key);
}


void DataLoader::create_cloud(std::string image_name, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr& cloud, pcl::PointCloud< pcl::PointXYZRGBA>::Ptr& cloud_unrectified) {
  cv::Mat depth = loadDepth(image_name);
  cv::Mat color = loadColor(image_name);
  Utils::Calibration calibration = loadCalibration(image_name);
  create_cloud(depth, color, calibration, cloud, cloud_unrectified);
}


void DataLoader::create_cloud(cv::Mat& depth, cv::Mat& color, Utils::Calibration& calibration, pcl::PointCloud< pcl::PointXYZRGBA>::Ptr& cloud, pcl::PointCloud< pcl::PointXYZRGBA>::Ptr& cloud_unrectified) {
  cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
  cloud_unrectified = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
  ushort* d_ptr = depth.ptr<ushort>(0);
  cloud->height = depth.rows;
  cloud->width = depth.cols;
  cloud->resize(cloud->height*cloud->width);
  cloud->is_dense = false;
  cloud_unrectified->height = depth.rows;
  cloud_unrectified->width = depth.cols;
  cloud_unrectified->resize(cloud_unrectified->height*cloud_unrectified->width);
  cloud_unrectified->is_dense = false;
  Eigen::MatrixXf mat(3, cloud->size()+1);
  unsigned int index = 0;
  for(unsigned int y = 0; y < cloud->height; y++){
    for(unsigned int x = 0; x  < cloud->width; x++){
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
  mat(0, index) = 0;
  mat(1, index) = 0;
  mat(2, index) = 0;

  Eigen::MatrixXf rect = (calibration._extrinsic.linear()* calibration._intrinsic_inverse)*mat + calibration._extrinsic.translation().rowwise().replicate(cloud->size()+1);
  Eigen::MatrixXf unrect =  calibration._intrinsic_inverse*mat;
  uchar* c_ptr = color.ptr<uchar>(0);
  uint8_t r, g, b;
  // uint32_t rgb;

  for(index = 0; index < cloud->size(); index++){
    cloud->points[index].x = rect(0,index);
    cloud->points[index].y = rect(1,index);
    cloud->points[index].z = rect(2,index);
    cloud_unrectified->points[index].x = unrect(0,index);
    cloud_unrectified->points[index].y = unrect(1,index);
    cloud_unrectified->points[index].z = unrect(2,index);
    b = c_ptr[0];
    g = c_ptr[1];
    r = c_ptr[2];
    c_ptr+=3;
    // rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    // cloud->points[index].rgb = *reinterpret_cast<float*>(&rgb);
    // cloud_unrectified->points[index].rgb = *reinterpret_cast<float*>(&rgb);
    cloud->points[index]. r = r;
    cloud->points[index]. g = g;
    cloud->points[index]. b = b;
    cloud_unrectified->points[index]. r = r;
    cloud_unrectified->points[index]. g = g;
    cloud_unrectified->points[index]. b = b;
  }
  cloud->sensor_origin_ = Eigen::Vector4f(rect(0, index),rect(1, index),rect(2, index),1);
  cloud_unrectified->sensor_origin_ = Eigen::Vector4f(0,0,0,1);
}


cv::Mat DataLoader::loadVoxel(std::string image_name) const{
  cv::Mat im = cv::imread(_voxel_dir + image_name + _voxel_ext);
  if(im.rows != 0 && im.cols != 0){
    return Utils::bgrToSegmentId(im);
  }else{
    throw std::runtime_error(std::string("Voxel image could not be loaded from ") + _voxel_dir + image_name + _voxel_ext);
  }
}


Utils::Calibration DataLoader::loadCalibration(std::string image_name) const{
  return Utils::Calibration(_calibration_dir + image_name + _calibration_ext);
}


cv::Mat DataLoader::loadLabel(std::string image_name) const{
  return _label_converter.rgbToLabel(cv::imread(_label_dir + image_name + _label_ext));
}


cv::Mat DataLoader::loadDepth(std::string image_name) const{
  return cv::imread(_depth_dir + image_name + _depth_ext, CV_LOAD_IMAGE_ANYDEPTH);
}


cv::Mat DataLoader::loadColor(std::string image_name) const{
  cv::Mat color = cv::imread(_color_dir + image_name + _color_ext);
  cv::cvtColor(color, color, CV_BGR2Lab);
  return color;
}


void DataLoader::extractVoxels(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud< pcl::PointXYZRGBA>::Ptr cloud_unrectifed, std::map< int, std::shared_ptr< Voxel> >& voxel_storage) {
  pcl::SupervoxelClustering<pcl::PointXYZRGBA> super (_voxel_resolution, _seed_resolution, true);
  super.setInputCloud(cloud_unrectifed);
  super.setColorImportance(_color_importance);
  super.setSpatialImportance(_spatial_importance);
  super.setNormalImportance(_normal_importance);
  std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > supervoxel_clusters;
  super.extract(supervoxel_clusters);

  //Doesn't seem to yield anything better really.
  //std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > refined_supervoxel_clusters;
  //super.refineSupervoxels(3, refined_supervoxel_clusters);

  pcl::PointCloud<pcl::PointXYZL>::Ptr full_labeled_cloud = super.getLabeledCloud();
  for(unsigned int j = 0; j < full_labeled_cloud->size(); ++j) {
    int label =  full_labeled_cloud->points[j].label;
    if(label > 0){ // this is the voxel containing all remaining points, we don't care about it.
      if(voxel_storage.count(label) == 0) {
        voxel_storage[label] = std::shared_ptr<Voxel>( new Voxel(cloud, label));
      }
      voxel_storage[label]->addPoint(j);
    }
  }
}

void DataLoader::extractVoxels(pcl::PointCloud< pcl::PointXYZRGBA>::Ptr cloud, std::map< int, std::shared_ptr< Voxel> >& voxel_storage,
                               pcl::PointCloud< pcl::PointXYZRGBA>::Ptr& voxelized_cloud) {
  pcl::SupervoxelClustering<pcl::PointXYZRGBA> super (_voxel_resolution, _seed_resolution, false);
  super.setInputCloud(cloud);
  super.setColorImportance(_color_importance);
  super.setSpatialImportance(_spatial_importance);
  super.setNormalImportance(_normal_importance);
  std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > supervoxel_clusters;
  super.extract(supervoxel_clusters);

  //Doesn't seem to yield anything better really.
  //std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGBA>::Ptr > refined_supervoxel_clusters;
  //super.refineSupervoxels(3, refined_supervoxel_clusters);

  voxelized_cloud = super.getColoredVoxelCloud();
  pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud();
  for(unsigned int j = 0; j < labeled_voxel_cloud->size(); ++j) {
    int label =  labeled_voxel_cloud->points[j].label;
    if(label > 0){ // this is the voxel containing all remaining points, we don't care about it.
      if(voxel_storage.count(label) == 0) {
        voxel_storage[label] = std::shared_ptr<Voxel>( new Voxel(voxelized_cloud, label, cloud));
      }
      voxel_storage[label]->addPoint(j);
    }
  }
  //Also collect the image based indices.
  pcl::PointCloud<pcl::PointXYZL>::Ptr full_labeled_cloud = super.getLabeledCloud();
  for(unsigned int j = 0; j < full_labeled_cloud->size(); ++j) {
    int label =  full_labeled_cloud->points[j].label;
    if(label > 0){ // this is the voxel containing all remaining points, we don't care about it.
      if(voxel_storage.count(label) != 0) {
        voxel_storage[label]->addImagePoint(j);
      }
    }
  }

}

void DataLoader::extractVoxelsFromImage(cv::Mat& voxel_image, pcl::PointCloud< pcl::PointXYZRGBA>::Ptr cloud, std::map< int, std::shared_ptr< Voxel > >& voxel_storage) {
  int* v_ptr = voxel_image.ptr<int>(0);
  for(int j = 0; j < voxel_image.rows*voxel_image.cols; ++j) {
    int v_label = v_ptr[j];
    if(v_label > 0){ // this is the voxel containing all remaining points, we don't care about it.
      if(voxel_storage.count(v_label) == 0) {
        voxel_storage[v_label] = std::shared_ptr<Voxel>( new Voxel(cloud, v_label));
      }
      voxel_storage[v_label]->addPoint(j);
    }
  }
}