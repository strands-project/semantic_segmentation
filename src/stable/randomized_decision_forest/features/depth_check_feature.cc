#include "depth_check_feature.hh"

using namespace Rdfs;

DepthCheckFeature::DepthCheckFeature(){
  m_feature_type=DEPTH_NAN_CHECK;
}

float DepthCheckFeature::ExtractFeature(const int x, const int y) const{
  if(isnan(m_image_depth_data[x+y*m_image_width])){
    return 1;
  }else{
    return -1;
  }
}

void DepthCheckFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
  m_image_width = data_image.Width();
}

BOOST_CLASS_EXPORT_IMPLEMENT(DepthCheckFeature)
