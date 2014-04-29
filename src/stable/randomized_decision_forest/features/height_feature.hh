#ifndef HEIGHT_FEATURE_HH
#define HEIGHT_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{
class HeightFeature : public TreeFeature
{
public:
  HeightFeature();

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
  }

private:
  Eigen::Matrix3f m_image_accelerometer_data; /// @brief The accelerometer data for the current image.
  float m_image_cx; /// @brief The cx camera parameter for the current image.
  float m_image_cy; /// @brief The cy camera parameter for the current image.
  float m_image_fx_inv; /// @brief The inverse of the fx camera parameter for the current image.
  float m_image_fy_inv; /// @brief The inverse of the fy camera parameter for the current image.
  const float * m_image_depth_data; /// @brief The pointer to the depth data for the current image.

};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::HeightFeature, "HeightFeature")
#endif // HEIGHT_FEATURE_HH
