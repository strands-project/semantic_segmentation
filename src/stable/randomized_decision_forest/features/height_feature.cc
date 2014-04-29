#include "height_feature.hh"


using namespace Rdfs;

HeightFeature::HeightFeature(){
  m_feature_type = HEIGHT;
}

float HeightFeature::ExtractFeature(const int x, const int y) const{
  float depth = m_image_depth_data[x+y*m_image_width];
  Eigen::Vector3f point;
  point << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * depth,
      (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * depth,
      depth;
  Eigen::Vector3f rectified = m_image_accelerometer_data * point;
  return rectified(1); //rectified height value;
}

void HeightFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_accelerometer_data = data_image.GetAccelerometerData();
  m_image_cx = data_image.m_cx_rgb;
  m_image_cy = data_image.m_cy_rgb;
  m_image_fx_inv = data_image.m_fx_rgb_inv;
  m_image_fy_inv = data_image.m_fy_rgb_inv;
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(HeightFeature)
