#pragma once

// libforest
#include "libforest/libforest.h"

// PCL includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/search/kdtree.h>


// STL includes
#include <vector>
#include <python2.7/grammar.h>
#include <cassert>


class Voxel{
public:
  Voxel(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr, int voxel_id):
      _cloud_ptr(cloud_ptr),
      _voxel_id(voxel_id),
      _label(-5),
      _has_features(false){
  }

  Voxel(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxel_ptr, int voxel_id, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr full_cloud_ptr):
      _cloud_ptr(voxel_ptr),
      _voxel_id(voxel_id),
      _label(-5),
      _has_features(false),
      _full_cloud_ptr(full_cloud_ptr){
  }

  ~Voxel(){}

  void addPoint(int idx){
    _point_indices.push_back(idx);
  }

  void addImagePoint(int idx){
    _image_point_indices.push_back(idx);
  }

  void computeLabel(cv::Mat label_image){
    std::vector<int> indices;
    if(_image_point_indices.size() != 0){
      indices = _image_point_indices;
    }else{
      indices = _point_indices;
    }
    std::map<int, int> label_count;
    int* l_ptr = label_image.ptr<int>(0);
    for(int index : indices){
      label_count[l_ptr[index]]++;
    }

    int top_count = -100;
    int top_label = -1; // We'll use -5 for not initialized or not computed.
    for(auto counts : label_count){
      if(counts.first >= 0){
        if(counts.second > top_count){
          top_count = counts.second;
          top_label = counts.first;
        }
      }
    }
    _label = top_label;
  }

  int getLabel() const{
    return _label;
  }

  int getSize() const{
    return _point_indices.size();
  }

  template<class T>
  void drawValueIntoImage(cv::Mat image, T value) const{
    std::vector<int> indices;
    if(_image_point_indices.size() != 0){
      indices = _image_point_indices;
    }else{
      indices = _point_indices;
    }
    T* i_ptr = image.ptr<T>(0);
    for(int p : indices){
      i_ptr[p] = value;
    }
  }

  void drawLabelIDIntoImage(cv::Mat image) const{
    drawValueIntoImage<int>(image, _label);
  }

  void drawSegmentIDIntoImage(cv::Mat image) const{
    drawValueIntoImage<int>(image, _voxel_id);
  }


  void drawProbIntoUnary(cv::Mat unary, const std::vector<float>& probability) const{
    std::vector<int> indices;
    if(_image_point_indices.size() != 0){
      indices = _image_point_indices;
    }else{
      indices = _point_indices;
    }

    const int dim = probability.size();
    std::vector<float> negative_log;
    negative_log.reserve(dim);
    for(float f : probability){
      negative_log.push_back(-log(f));
    }

    float* ptr = unary.ptr<float>(0);
    for(int p :indices){
      float* p_ptr = ptr + p*dim;
      for(float f : negative_log){
        *p_ptr++ = f;
      }
    }
  }

  void addDataToCrfMats(Eigen::MatrixXf& unary,
                        Eigen::MatrixXf& pairwise1,
                        Eigen::MatrixXf& pairwise2,
                        uint& index,
                        std::vector<float>& probability,
                        float appearance_color_sigma,
                        float appearance_range_sigma,
                        float smoothness_range_sigma) const{
    const int dim = probability.size();
    std::vector<float> negative_log;
    negative_log.reserve(dim);
    for(float f : probability){
      negative_log.push_back(-log(f));
    }

    //for each point
    for(int p :_point_indices){
      //Add the unary
      for(int i = 0; i < dim; i++){
        unary(i, index) = negative_log[i];
      }

      //Add the 3D location to both features.
      pairwise1(0, index) = _cloud_ptr->points[p].x / appearance_range_sigma;
      pairwise1(1, index) = _cloud_ptr->points[p].y / appearance_range_sigma;
      pairwise1(2, index) = _cloud_ptr->points[p].z / appearance_range_sigma;
      pairwise2(0, index) = _cloud_ptr->points[p].x / smoothness_range_sigma;
      pairwise2(1, index) = _cloud_ptr->points[p].y / smoothness_range_sigma;
      pairwise2(2, index) = _cloud_ptr->points[p].z / smoothness_range_sigma;

      //Add the color part to the first pairwise
      pairwise1(3, index) = _cloud_ptr->points[p].r / appearance_color_sigma;
      pairwise1(4, index) = _cloud_ptr->points[p].g / appearance_color_sigma;
      pairwise1(5, index) = _cloud_ptr->points[p].b / appearance_color_sigma;

      //Increment the index for the next point
      index++;
    }
  }
#define BINS 15
#define FEATURE_LENGTH 59

  void computeFeatures(bool use_full = true){
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cld;
    std::vector<int> indices;
    if(use_full && _image_point_indices.size() != 0){
      indices = _image_point_indices;
      cld = _full_cloud_ptr;
    }else{
      indices = _point_indices;
      cld = _cloud_ptr;
    }

  void computeFeatures(){
    float x = 0, y = 0, z = 0;
    float r = 0,g =0 ,b =0, rr =0, gg =0, bb =0;
    for(int p : _point_indices){
      x += _cloud_ptr->points[p].x;
      y += _cloud_ptr->points[p].y;
      z += _cloud_ptr->points[p].z;
      r += _cloud_ptr->points[p].r;
      g += _cloud_ptr->points[p].g;
      b += _cloud_ptr->points[p].b;
      rr += _cloud_ptr->points[p].r*_cloud_ptr->points[p].r;
      gg += _cloud_ptr->points[p].g*_cloud_ptr->points[p].g;
      bb += _cloud_ptr->points[p].b*_cloud_ptr->points[p].b;
    }
    float count_inv = 1.0f/static_cast<float>(_point_indices.size());
    r *= count_inv;
    g *= count_inv;
    b *= count_inv;
    rr *= count_inv;
    gg *= count_inv;
    bb *= count_inv;
    x *= count_inv;
    y *= count_inv;
    z *= count_inv;

    float sigma_p;
    float sigma_s;
    float sigma_l;
    float curvature;
    float tah;
    float nah;
    float bb1;
    float bb2;
    float bb3;

    Eigen::Vector4f origin = cld->sensor_origin_;
    Eigen::Vector4f centroid;
    centroid(0) = x;
    centroid(1) = y;
    centroid(2) = z;
    centroid(3) = 1;

    std::vector<float> features;
    Eigen::Matrix3f covariance_matrix;
    pcl::computeCovarianceMatrix(*(cld), indices, centroid, covariance_matrix);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(covariance_matrix);

    sigma_p = es.eigenvalues()(0) ; // was (2) at some point, but (0) is the smallest which makes the most sense!
    sigma_l = es.eigenvalues()(2) - es.eigenvalues()(1);
    sigma_s = es.eigenvalues()(1) - es.eigenvalues()(0);
    if(isnan(sigma_l) || isnan(sigma_p) || isnan(sigma_s)){
      std::cout << "oohoho" << std::endl;
    }

    curvature = es.eigenvalues()(0) / (es.eigenvalues()(0) + es.eigenvalues()(1) + es.eigenvalues()(2));

    Eigen::Vector3f vn = es.eigenvectors().col(0);
    Eigen::Vector3f vt = es.eigenvectors().col(2);

    // Flip the plane normal if needed
    //This should be origin-centroid, but it makes the normals point downwards?
    //Likely this is due to the fact that everything is negative regarding y direction.
    Eigen::Vector4f vp = centroid - origin;
    Eigen::Vector3f vp3;
    vp3 << vp(0), vp(1), vp(2);
    if (vp3.dot (vn) < 0) {
      vn *= -1.0;
    }

    nah = (vn(2) / vn.norm()) * (sigma_l / std::max(sigma_l, std::max(sigma_p, sigma_s)));
    tah = (vt(2) / vt.norm()) * (sigma_s / std::max(sigma_l, std::max(sigma_p, sigma_s)));


    float bbx_min = std::numeric_limits<float>::max();
    float bbx_max = std::numeric_limits<float>::max() * -1.0;
    float bby_min = std::numeric_limits<float>::max();
    float bby_max = std::numeric_limits<float>::max() * -1.0;
    float bbz_min = std::numeric_limits<float>::max();
    float bbz_max = std::numeric_limits<float>::max() * -1.0;
    const Eigen::Vector3f Dx = es.eigenvectors().col(0);
    const Eigen::Vector3f Dy = es.eigenvectors().col(1);
    const Eigen::Vector3f Dz = es.eigenvectors().col(2);
    for (uint j = 0; j < indices.size(); j++) {
      Eigen::Vector3f W(cld->points[indices[j]].x,
                        cld->points[indices[j]].y,
                        cld->points[indices[j]].z);


      float tmpx = W.dot(Dx);
      float tmpy = W.dot(Dy);
      float tmpz = W.dot(Dz);

      if (tmpx < bbx_min) {
        bbx_min = tmpx;
      }
      if (tmpx > bbx_max) {
        bbx_max = tmpx;
      }

      if (tmpy < bby_min) {
        bby_min = tmpy;
      }
      if (tmpy > bby_max) {
        bby_max = tmpy;
      }

      if (tmpz < bbz_min) {
        bbz_min = tmpz;
      }
      if (tmpz > bbz_max) {
        bbz_max = tmpz;
      }
    }


    bb1 = bbx_max - bbx_min;
    bb2 = bby_max - bby_min;
    bb3 = bbz_max - bbz_min;


#define FEATURE_LENGTH 20

    _features = libf::DataPoint(FEATURE_LENGTH);
    int f_index = 0;
    _features(f_index++) = z;
    _features(f_index++) = r;
    _features(f_index++) = g;
    _features(f_index++) = b;
    _features(f_index++) = rr - r*r;
    _features(f_index++) = gg - g*g ;
    _features(f_index++) = bb - b*b;
    _features(f_index++) = sigma_p;
    _features(f_index++) = sigma_s;
    _features(f_index++) = sigma_l;
    _features(f_index++) = curvature;
    _features(f_index++) = vn(0) / vn.norm();
    _features(f_index++) = vn(1) / vn.norm();
    _features(f_index++) = vn(2) / vn.norm();
    _features(f_index++) = acos(vn(2) / vn.norm()); //angle between normal and up vector.
    _features(f_index++) = nah;
    _features(f_index++) = tah;
    _features(f_index++) = bb1;
    _features(f_index++) = bb2;
    _features(f_index++) = bb3;
    for(int i = 0; i < FEATURE_LENGTH; i++){
      if(isnan(_features(i))){
        std::cout << i << std::endl;
      }
    }

    _has_features = true;
  }

  const libf::DataPoint& getFeatures(){
    if(!_has_features){
      computeFeatures();
    }
    return _features;
  }

  int getVoxelID() const{
    return _voxel_id;
  }

  static int getFeatureSize(){
    return FEATURE_LENGTH;
  }

  const std::vector<int>& getIndices() const{
    return _point_indices;
  }

private:
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr _cloud_ptr;
  int _voxel_id;
  signed char  _label;
  bool _has_features;

  std::vector<int> _image_point_indices;
  std::vector<int> _point_indices;
  std::vector<float> _class_distrubition;
  libf::DataPoint _features;

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr _full_cloud_ptr;
};