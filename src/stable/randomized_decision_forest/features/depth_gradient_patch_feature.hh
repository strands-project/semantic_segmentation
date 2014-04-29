#ifndef DEPTH_GRADIENT_PATCH_FEATURE_HH
#define DEPTH_GRADIENT_PATCH_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{

class DepthGradientPatchFeature : public TreeFeature
{
public:
  DepthGradientPatchFeature(int x1=0, int y1=0, int c1=0, int x2=0, int y2=0, bool scale_with_depth = false);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_x1;
    ar & m_y1;
    ar & m_c;
    ar & m_x2;
    ar & m_y2;
    ar & m_scale_with_depth;
  }

private:
  int m_x1; /// @brief x_offset for corner pixel 1
  int m_y1; /// @brief y_offset for corner pixel 1
  int m_c;  /// @brief color channel for pixel 1
  int m_x2; /// @brief x_offset for corner pixel 2 seen from the oposite corner
  int m_y2; /// @brief y_offset for corner pixel 2 seen from the oposite corner
  bool m_scale_with_depth; /// @brief Set to True if the patches of this feature should be scaled with the depth.
  const float * m_image_depth_data; /// @brief Pointer to the depth data of the current image.
  const float * m_image_depth_gradient_data; /// @brief Pointer to the depth gradient data of the current image.
  int m_image_depth_gradient_bins; /// @brief Number of bins in the depth gradient data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::DepthGradientPatchFeature, "DepthGradientPatchFeature")


#endif // DEPTH_GRADIENT_PATCH_FEATURE_HH
