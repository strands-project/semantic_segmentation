#include "color_feature_addition.hh"

using namespace Rdfs;

ColorFeatureAddition::ColorFeatureAddition(int x1, int y1, int c1, int x2, int y2, int c2):
  m_x1(x1), m_y1(y1), m_c1(c1), m_x2(x2), m_y2(y2), m_c2(c2){
  m_feature_type = COLOR_ADD;
}

float ColorFeatureAddition::ExtractFeature(const int x, const int y) const{
  int x1 = x + m_x1;
  int y1 = y + m_y1;
  x1 = std::min(x1, m_image_width-1);
  y1 = std::min(y1, m_image_height -1);
  x1 = std::max(x1, 0);
  y1 = std::max(y1, 0);

  int x2 = x + m_x2;
  int y2 = y + m_y2;
  x2 = std::min(x2, m_image_width-1);
  y2 = std::min(y2, m_image_height -1);
  x2 = std::max(x2, 0);
  y2 = std::max(y2, 0);
  return m_image_lab_data[3 * (m_image_width * y1 + x1) + m_c1] +  m_image_lab_data[3 * (m_image_width * y2 + x2) + m_c2];
  //return (data_image.GetLABImage())(x1, y1, m_c1) + (data_image.GetLABImage())(x2, y2, m_c2);
}

void ColorFeatureAddition::InitEvaluation(const Utils::DataImage &data_image){
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_lab_data = data_image.GetLABImage().ptr<unsigned char>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(ColorFeatureAddition)
