#include "image_coordinate_feature.hh"

using namespace Rdfs;

ImageCoordinateFeature::ImageCoordinateFeature(int type){
  assert(type==X_PIXEL_POS || type==Y_PIXEL_POS || type ==-1); // type == -1 needed for init of tree when it is loaded!
  m_feature_type = type;
}

float ImageCoordinateFeature::ExtractFeature(const int x, const int y) const{
  if(m_feature_type==X_PIXEL_POS){
    assert(x>=0 && x < m_image_width);
    return static_cast<float>(x)/static_cast<float>(m_image_width);
  }else{
    assert(y>=0 && y < m_image_height);
    return static_cast<float>(y)/static_cast<float>(m_image_height);
  }
}

void ImageCoordinateFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
}

BOOST_CLASS_EXPORT_IMPLEMENT(ImageCoordinateFeature)
