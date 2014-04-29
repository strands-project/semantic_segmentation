#ifndef ORDINAL_DEPTH_CHECK_FEATURE_HH
#define ORDINAL_DEPTH_CHECK_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{

class OrdinalDepthFeature : public TreeFeature
{
public:
  OrdinalDepthFeature(int max_index=0, std::vector<int> x_offsets=std::vector<int>(0), std::vector<int> y_offsets=std::vector<int>(0));

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_max_index;
    ar & m_x_offsets;
    ar & m_y_offsets;
  }

private:
  int m_max_index; /// @brief index for the max valued pixel.
  std::vector<int> m_x_offsets; /// @brief x_offset for pixel 1
  std::vector<int> m_y_offsets; /// @brief y_offset for pixel 1
  const float * m_image_depth_data; /// @brief Pointer to the depth data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::OrdinalDepthFeature, "OrdinalDepthFeature")

#endif // ORDINAL_DEPTH_CHECK_FEATURE_HH
