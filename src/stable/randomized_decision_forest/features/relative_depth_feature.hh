#ifndef RELATIVE_DEPTH_FEATURE_HH
#define RELATIVE_DEPTH_FEATURE_HH

// Local include
#include "tree_feature.hh"

namespace Rdfs{

class RelativeDepthFeature : public TreeFeature
{
public:
  RelativeDepthFeature(int c=0);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_c;
  }

private:
  int m_c; /// @brief Decides which channel to use, either the original or the non normalized.
  const float * m_image_normalization_data; /// @brief Pointer to the normalization data of the current image.
  const float * m_image_depth_data;         /// @brief Pointer to the depth data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::RelativeDepthFeature, "RelativeDepthFeature")
#endif // RELATIVE_DEPTH_FEATURE_HH
