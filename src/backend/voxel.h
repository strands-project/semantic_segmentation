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

  void addNeighborPoints(const std::vector<int>& indices){
    _neighbor_points.insert(_neighbor_points.end(), indices.begin(), indices.end());
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
#define BINS 17
#define FEATURE_LENGTH 14

  void computeFeatures(){
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cld;
    std::vector<int> indices;
    if(_image_point_indices.size() != 0){
      indices = _image_point_indices;
      cld = _full_cloud_ptr;
    }else{
      indices = _point_indices;
      cld = _cloud_ptr;
    }

    float x = 0, y = 0, z = 0;
    std::vector<float> hist(BINS*3,0);
    for(int p : indices){
      x += cld->points[p].x;
      y += cld->points[p].y;
      z += cld->points[p].z;
      hist[std::min(cld->points[p].r/BINS, BINS)]++;
      hist[std::min(cld->points[p].g/BINS + BINS, BINS)]++;
      hist[std::min(cld->points[p].b/BINS + 2*BINS, BINS)]++;
    }
    float count_inv = 1.0f/static_cast<float>(indices.size());
    for(int i = 0; i < 3*BINS; ++i){
      hist[i] *= count_inv;
    }

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

      bbx_min = std::min(tmpx, bbx_min);
      bbx_max = std::max(tmpx, bbx_max);
      bby_min = std::min(tmpy, bby_min);
      bby_max = std::max(tmpy, bby_max);
      bbz_min = std::min(tmpz, bbz_min);
      bbz_max = std::max(tmpz, bbz_max);
    }


    bb1 = bbx_max - bbx_min;
    bb2 = bby_max - bby_min;
    bb3 = bbz_max - bbz_min;

    int total_length = FEATURE_LENGTH + 3*BINS;

    if(_neighbor_points.size() > 0){
      total_length *=2;
      indices.insert(indices.end(), _neighbor_points.begin(), _neighbor_points.end());
    }




    _features = libf::DataPoint(total_length);
    int f_index = 0;
    _features(f_index++) = z;
    for(float h : hist){
      _features(f_index++) = h;
    }
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



    if(_neighbor_points.size() > 0){
      float x_n = 0, y_n = 0, z_n = 0;
      std::vector<float> hist_n(BINS*3,0);
      for(int p : indices){
        x_n += cld->points[p].x;
        y_n += cld->points[p].y;
        z_n += cld->points[p].z;
        hist_n[std::min(cld->points[p].r/BINS, BINS)]++;
        hist_n[std::min(cld->points[p].g/BINS + BINS, BINS)]++;
        hist_n[std::min(cld->points[p].b/BINS + 2*BINS, BINS)]++;
      }
      float count_inv = 1.0f/static_cast<float>(indices.size());
      for(int i = 0; i < 3*BINS; ++i){
        hist_n[i] *= count_inv;
      }

      x_n *= count_inv;
      y_n *= count_inv;
      z_n *= count_inv;

      float sigma_p_n;
      float sigma_s_n;
      float sigma_l_n;
      float curvature_n;
      float tah_n;
      float nah_n;
      float bb1_n;
      float bb2_n;
      float bb3_n;

      Eigen::Vector4f origin_n = cld->sensor_origin_;
      Eigen::Vector4f centroid_n;
      centroid_n(0) = x_n;
      centroid_n(1) = y_n;
      centroid_n(2) = z_n;
      centroid_n(3) = 1;

      std::vector<float> features_n;
      Eigen::Matrix3f covariance_matrix_n;
      pcl::computeCovarianceMatrix(*(cld), indices, centroid_n, covariance_matrix_n);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es_n(covariance_matrix_n);

      sigma_p_n = es_n.eigenvalues()(0) ; // was (2) at some point, but (0) is the smallest which makes the most sense!
      sigma_l_n = es_n.eigenvalues()(2) - es_n.eigenvalues()(1);
      sigma_s_n = es_n.eigenvalues()(1) - es_n.eigenvalues()(0);
      if(isnan(sigma_l_n) || isnan(sigma_p_n) || isnan(sigma_s_n)){
        std::cout << "oohoho" << std::endl;
      }

      curvature_n = es_n.eigenvalues()(0) / (es_n.eigenvalues()(0) + es_n.eigenvalues()(1) + es_n.eigenvalues()(2));

      Eigen::Vector3f vn_n = es_n.eigenvectors().col(0);
      Eigen::Vector3f vt_n = es_n.eigenvectors().col(2);

      // Flip the plane normal if needed
      //This should be origin-centroid, but it makes the normals point downwards?
      //Likely this is due to the fact that everything is negative regarding y direction.
      Eigen::Vector4f vp_n = centroid_n - origin_n;
      Eigen::Vector3f vp3_n;
      vp3_n << vp_n(0), vp_n(1), vp_n(2);
      if (vp3_n.dot (vn_n) < 0) {
        vn_n *= -1.0;
      }

      nah_n = (vn_n(2) / vn_n.norm()) * (sigma_l_n / std::max(sigma_l_n, std::max(sigma_p_n, sigma_s_n)));
      tah_n = (vt_n(2) / vt_n.norm()) * (sigma_s_n / std::max(sigma_l_n, std::max(sigma_p_n, sigma_s_n)));


      float bbx_min_n = std::numeric_limits<float>::max();
      float bbx_max_n = std::numeric_limits<float>::max() * -1.0;
      float bby_min_n = std::numeric_limits<float>::max();
      float bby_max_n = std::numeric_limits<float>::max() * -1.0;
      float bbz_min_n = std::numeric_limits<float>::max();
      float bbz_max_n = std::numeric_limits<float>::max() * -1.0;
      const Eigen::Vector3f Dx_n = es_n.eigenvectors().col(0);
      const Eigen::Vector3f Dy_n = es_n.eigenvectors().col(1);
      const Eigen::Vector3f Dz_n = es_n.eigenvectors().col(2);
      for (uint j = 0; j < indices.size(); j++) {
        Eigen::Vector3f W_n(cld->points[indices[j]].x,
                          cld->points[indices[j]].y,
                          cld->points[indices[j]].z);


        float tmpx_n = W_n.dot(Dx_n);
        float tmpy_n = W_n.dot(Dy_n);
        float tmpz_n = W_n.dot(Dz_n);

        bbx_min_n = std::min(tmpx_n, bbx_min_n);
        bbx_max_n = std::max(tmpx_n, bbx_max_n);
        bby_min_n = std::min(tmpy_n, bby_min_n);
        bby_max_n = std::max(tmpy_n, bby_max_n);
        bbz_min_n = std::min(tmpz_n, bbz_min_n);
        bbz_max_n = std::max(tmpz_n, bbz_max_n);
      }


      bb1_n = bbx_max_n - bbx_min_n;
      bb2_n = bby_max_n - bby_min_n;
      bb3_n = bbz_max_n - bbz_min_n;


      _features(f_index++) = z_n;
      for(float h : hist_n){
        _features(f_index++) = h;
      }
      _features(f_index++) = sigma_p_n;
      _features(f_index++) = sigma_s_n;
      _features(f_index++) = sigma_l_n;
      _features(f_index++) = curvature_n;
      _features(f_index++) = vn_n(0) / vn_n.norm();
      _features(f_index++) = vn_n(1) / vn_n.norm();
      _features(f_index++) = vn_n(2) / vn_n.norm();
      _features(f_index++) = acos(vn_n(2) / vn_n.norm()); //angle between normal and up vector.
      _features(f_index++) = nah_n;
      _features(f_index++) = tah_n;
      _features(f_index++) = bb1_n;
      _features(f_index++) = bb2_n;
      _features(f_index++) = bb3_n;
    }

    for(int i = 0; i < total_length; i++){
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

  int getFeatureSize() const{
    return _features.size();
  }

  const std::vector<int>& getIndices() const{
    return _point_indices;
  }

  const std::vector<int>& getFullCloudIndices() const{
    return _image_point_indices;
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

  std::vector<int>> _neighbor_points;
};