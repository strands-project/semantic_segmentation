#ifndef NORMAL_FEATURE_HH
#define NORMAL_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{
class NormalFeature : public TreeFeature
{
public:
  NormalFeature(int x1=0, int y1=0, int x2=0, int y2=0, int c1=0, Eigen::Vector3f vec =Eigen::Vector3f(0.0f,0.0f,0.0f));

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_c;
	 ar & m_x1;
	 ar & m_y1;
	 ar & m_x2;
	 ar & m_y2;
   ar & m_vec_0;
   ar & m_vec_1;
   ar & m_vec_2;
  }

private:
  int m_x1; /// @brief x_offset for pixel 1
  int m_y1; /// @brief y_offset for pixel 1
  int m_x2; /// @brief x_offset for pixel 2
  int m_y2; /// @brief y_offset for pixel 2
  int m_c;  /// @brief channel for this feature.
  float m_vec_0; /// @brief stores the vector a normal will be compared to.
  float m_vec_1; ///        This is done in three floats because of the serialization.
  float m_vec_2;
  Eigen::Matrix3f m_image_accelerometer_data; /// @brief The accelerometer data for the current image. Will be inversed.
  float m_image_cx; /// @brief The cx camera parameter for the current image.
  float m_image_cy; /// @brief The cy camera parameter for the current image.
  float m_image_fx_inv; /// @brief The inverse of the fx camera parameter for the current image.
  float m_image_fy_inv; /// @brief The inverse of the fy camera parameter for the current image.
  const float * m_image_normal_data; /// @brief The pointer to the normal data for the current image.
  const float * m_image_depth_data; /// @brief Pointer to the depth data of the current image.
  Eigen::Vector3f m_vector; /// @brief an actual vector, created from the three m_vec_# entries.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::NormalFeature, "NormalFeature")
#endif // NORMAL_FEATURE_HH
