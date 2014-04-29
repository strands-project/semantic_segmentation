#ifndef DEPTH_COMPARISON_FEATURE_HH
#define DEPTH_COMPARISON_FEATURE_HH

// Local includes
#include "tree_feature.hh"

namespace Rdfs{

class DepthComparisonFeature : public TreeFeature
{
public:
  DepthComparisonFeature(int x1=0, int y1=0, int x2=0, int y2=0);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_x1;
    ar & m_y1;
    ar & m_x2;
    ar & m_y2;
  }

private:
  int m_x1; /// @brief x_offset for pixel 1
  int m_y1; /// @brief y_offset for pixel 1
  int m_x2; /// @brief x_offset for pixel 2
  int m_y2; /// @brief y_offset for pixel 2
  const float * m_image_depth_data; /// @brief Pointer to the depth data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::DepthComparisonFeature, "DepthComparisonFeature")

#endif // DEPTH_COMPARISON_FEATURE_HH
