#include "relative_depth_feature.hh"

using namespace Rdfs;

RelativeDepthFeature::RelativeDepthFeature(int c):m_c(c){
  m_feature_type = RELATIVE_DEPTH;
}

float RelativeDepthFeature::ExtractFeature(const int x, const int y) const{
  if(m_c==0){
    return m_image_depth_data[x+y*m_image_width] * m_image_normalization_data[x];
  }else{
    return 1.0f/m_image_normalization_data[x] - m_image_depth_data[x+y*m_image_width];
  }
}

void RelativeDepthFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_normalization_data = data_image.GetNormalizationData();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(RelativeDepthFeature)
