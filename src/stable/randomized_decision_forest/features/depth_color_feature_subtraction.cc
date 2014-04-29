#include "depth_color_feature_subtraction.hh"

using namespace Rdfs;

DepthColorFeatureSubtraction::DepthColorFeatureSubtraction(int x1, int y1, int c1, int x2, int y2, int c2):
  m_x1(x1), m_y1(y1), m_c1(c1), m_x2(x2), m_y2(y2), m_c2(c2){
  m_feature_type = HYBRID_SUB;
}

float DepthColorFeatureSubtraction::ExtractFeature(const int x, const int y) const{
  float current = 1.0f/m_image_depth_data[x+y*m_image_width];
  int x1 = x+ static_cast<float>(m_x1) * current +0.5f;
  int y1 = y+ static_cast<float>(m_y1) * current +0.5f;
  x1 = std::min(x1, m_image_width-1);
  y1 = std::min(y1, m_image_height -1);
  x1 = std::max(x1, 0);
  y1 = std::max(y1, 0);

  int x2 = x+ static_cast<float>(m_x2) * current +0.5f;
  int y2 = y+ static_cast<float>(m_y2) * current +0.5f;
  x2 = std::min(x2, m_image_width-1);
  y2 = std::min(y2, m_image_height -1);
  x2 = std::max(x2, 0);
  y2 = std::max(y2, 0);
  return m_image_lab_data[3*(x1+m_image_width*y1)+m_c1] - m_image_lab_data[3*(x2+m_image_width*y2)+m_c2];
}

void DepthColorFeatureSubtraction::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
  m_image_lab_data = data_image.GetLABImage().ptr<unsigned char>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(DepthColorFeatureSubtraction)
