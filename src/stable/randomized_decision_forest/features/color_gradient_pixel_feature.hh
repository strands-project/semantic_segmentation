#ifndef COLOR_GRADIENT_PIXEL_FEATURE_HH
#define COLOR_GRADIENT_PIXEL_FEATURE_HH

//Local includes
#include "tree_feature.hh"

namespace Rdfs{

class ColorGradientPixelFeature : public TreeFeature
{
public:
  ColorGradientPixelFeature(int x1=0, int y1=0, int c1=0, int x2=0, int y2=0, int c2=0);

  float ExtractFeature(const int x, const int y) const;

  void InitEvaluation(const Utils::DataImage &data_image) ;


private:
  friend class boost::serialization::access;
  template <class Archive> void serialize(Archive &ar, const unsigned int version) {
    ar & boost::serialization::base_object<TreeFeature>(*this);
    ar & m_x1;
    ar & m_y1;
    ar & m_c1;
    ar & m_x2;
    ar & m_y2;
    ar & m_c2;
  }

private:
  int m_x1; /// @brief x_offset for pixel 1
  int m_y1; /// @brief y_offset for pixel 1
  int m_c1; /// @brief color channel for pixel 1
  int m_x2; /// @brief x_offset for pixel 2
  int m_y2; /// @brief y_offset for pixel 2
  int m_c2; /// @brief color channel for pixel 2
  const float* m_image_gradient_data; /// @brief Pointer to the color gradient data of the current image.
  int m_image_gradient_bins; /// @brief Number of bins in the gradient data of the current image.
};
}

BOOST_CLASS_EXPORT_KEY2(Rdfs::ColorGradientPixelFeature, "ColorGradientPixelFeature")

#endif // COLOR_GRADIENT_PIXEL_FEATURE_HH
