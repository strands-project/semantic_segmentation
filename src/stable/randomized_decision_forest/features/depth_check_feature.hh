#ifndef DEPTH_CHECK_FEATURE_HH
#define DEPTH_CHECK_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{

class DepthCheckFeature : public TreeFeature
{
public:
  DepthCheckFeature();

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
  }

private:
  const float * m_image_depth_data; /// @brief Pointer to the depth data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::DepthCheckFeature, "DepthCheckFeature")

#endif // DEPTH_CHECK_FEATURE_HH
