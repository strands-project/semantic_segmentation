#include "single_pixel_color_feature.hh"

using namespace Rdfs;

ColorFeature::ColorFeature(int x, int y, int c): m_x(x), m_y(y), m_c(x){
  m_feature_type = PIXEL_COLOR;
}

float ColorFeature::ExtractFeature(const int x, const int y) const{
  int x1 = x + m_x;
  int y1 = y + m_y;
  x1 = std::min(x1, m_image_width-1);
  y1 = std::min(y1, m_image_height -1);
  x1 = std::max(x1, 0);
  y1 = std::max(y1, 0);
  return m_image_lab_data[3*(x1+m_image_width*y1)+m_c]; 
}

void ColorFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_lab_data = data_image.GetLABImage().ptr<unsigned char>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(ColorFeature)
