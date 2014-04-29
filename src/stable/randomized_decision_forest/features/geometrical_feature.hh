#ifndef GEOMETRICAL_FEATURE_HH
#define GEOMETRICAL_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{

class GeometricalFeature : public TreeFeature
{
public:
  GeometricalFeature(int c=0);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_c;
  }

private:
  int m_c; /// @brief feature channel for pixel 1
  const float * m_image_geometrical_data; /// @brief Pointer to the geometrical feature data of the current image.
  int m_image_geometrical_feature_count; /// @brief Number of different geometrical features stored for the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::GeometricalFeature, "GeometricalFeature")

#endif // GEOMETRICAL_FEATURE_HH
