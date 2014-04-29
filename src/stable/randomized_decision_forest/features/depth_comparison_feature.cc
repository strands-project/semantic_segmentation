#include "depth_comparison_feature.hh"

using namespace Rdfs;

DepthComparisonFeature::DepthComparisonFeature(int x1, int y1, int x2, int y2):
  m_x1(x1), m_y1(y1), m_x2(x2), m_y2(y2){
  m_feature_type = DEPTH_FEAT;
}

float DepthComparisonFeature::ExtractFeature(const int x, const int y) const{
  float current = 1.0f/m_image_depth_data[x+y*m_image_width];
  int x1, x2, y1, y2;
  x1 = x+ (static_cast<float>(m_x1) * current +0.5f);
  x2 = x+ (static_cast<float>(m_x2) * current +0.5f);
  y1 = y+ (static_cast<float>(m_y1) * current +0.5f);
  y2 = y+ (static_cast<float>(m_y2) * current +0.5f);

  float f1= 10.0f;
  float f2= 10.0f;

  if((x1 >=0) ){
    if( x1 < m_image_width){
      if(y1 >=0 ){
        if( y1 < m_image_height){
          f1 = m_image_depth_data[x1+y1*m_image_width];
        }
      }
    }
  }

  if((x2 >=0) ){
    if( x2 < m_image_width){
      if(y2 >=0 ){
        if( y2 < m_image_height){
          f2 = m_image_depth_data[x2+y2*m_image_width];
        }
      }
    }
  }
  if(isnan(f1)){
    f1= 10.0f;
  }

  if(isnan(f2)){
    f2= 10.0f;
  }
  return f1-f2;

}

void DepthComparisonFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(DepthComparisonFeature)
