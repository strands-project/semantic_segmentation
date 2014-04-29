#ifndef IMAGE_COORDINATE_FEATURE_HH
#define IMAGE_COORDINATE_FEATURE_HH

// Local includes
#include "tree_feature.hh"

namespace Rdfs{

class ImageCoordinateFeature : public TreeFeature
{
public:
  ImageCoordinateFeature(int type=-1);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
  }
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::ImageCoordinateFeature, "ImageCoordinateFeature")

#endif // IMAGE_COORDINATE_FEATURE_HH
