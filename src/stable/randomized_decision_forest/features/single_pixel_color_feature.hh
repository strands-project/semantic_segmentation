#ifndef SINGLE_PIXEL_COLORFEATURE_HH
#define SINGLE_PIXEL_COLORFEATURE_HH

#include "tree_feature.hh"


namespace Rdfs {

class ColorFeature : public TreeFeature
{
public:
  ColorFeature(int x=0, int y=0, int c=0);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_x;
    ar & m_y;
    ar & m_c;
  }

private:
  int m_x; /// @brief x offset
  int m_y; /// @brief y_offset
  int m_c; /// @brief color chanel
  const unsigned char* m_image_lab_data; /// @brief Pointer to the LAB data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::ColorFeature, "ColorFeature")

#endif // SINGLE_PIXEL_COLORFEATURE_HH
