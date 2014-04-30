#ifndef DENSECRF_EVALUATOR_HH
#define DENSECRF_EVALUATOR_HH

// Graphics includes
//#include <graphics/image.hh> <- This needs to be removed once we want to do the reconstruction again!

// Utils includes
#include <utils/data_image.hh>

//Local includes
#include "cluster3d.hh"

//STL includes
#include <deque>
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace Inference{
class DenseCRFEvaluator{
public:
  DenseCRFEvaluator();
  DenseCRFEvaluator(std::string config_filename);

  ~DenseCRFEvaluator();

  void Evaluate(const Utils::DataImage &image, cv::Mat &segmentation_result) const;
 // void Evaluate(const std::deque<Utils::DataImage> &images, const std::vector<Cluster3D> &cluster_data, int frame_index, Graphics::Image<signed char> &segmentation_result) const;
 // void Evaluate3D(const std::deque<Utils::DataImage> &images, std::vector<Cluster3D> &cluster_data, int frame_index, Graphics::Image<signed char> &segmentation_result) const;
  void Evaluate3D(std::vector<Cluster3D> &cluster_data, int frame_index) const;
//  void Evaluate2D(const Utils::DataImage &image, Graphics::Image<float> &segmentation_result) const;

  void StopParallelization();

  int GetLoadRequirements();

private:
  float m_smoothing_weight;
  float m_color_term_weight;
  float m_depth_term_weight;
  float m_pixel_sigma;
  float m_longe_range_pixel_sigma; //Used for the color term. 
  float m_depth_sigma;
  float m_depth_sigma_long_range;
  float m_color_sigma;
  float m_depth_color_sigma;
  float m_normal_sigma;
  float m_iteration_count;
  bool m_use_normals;
  int m_class_count;

  bool m_use_consistency_step;
  bool m_use_center;
  int m_consistency_window;
  int m_consistency_window_subsampling;

  bool m_do_not_parallelize;

};
}
#endif // DENSECRF_EVALUATOR_HH
