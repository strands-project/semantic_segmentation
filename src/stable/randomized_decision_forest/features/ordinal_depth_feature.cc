#include "ordinal_depth_feature.hh"

#include <utility>

using namespace Rdfs;

bool pairCompare(const std::pair<float, int>& firstElem, const std::pair<float, int>& secondElem) {
  return firstElem.first < secondElem.first;

}

OrdinalDepthFeature::OrdinalDepthFeature(int max_index, std::vector<int> x_offsets, std::vector<int> y_offsets):
  m_max_index(max_index), m_x_offsets(x_offsets), m_y_offsets(y_offsets){
  m_feature_type=ORDINAL_DEPTH;
}

float OrdinalDepthFeature::ExtractFeature(const int x, const int y) const{
  //Get current depth for offset scaling.
  float current = 1.0f/m_image_depth_data[x+y*m_image_width];

  //Setup a variable for the depth values at different offsets.
  int n = m_x_offsets.size();
  std::vector<std::pair<float, int> > res(n);

  //Compute results.
  int x1, y1;
  float f1 = 0.0f;
  for(unsigned int i=0; i < n; i++){
    x1 = x + (static_cast<float>(m_x_offsets[i]) * current +0.5f);
    y1 = y + (static_cast<float>(m_y_offsets[i]) * current +0.5f);

    if((x1 >=0) ){
      if( x1 < m_image_width){
        if(y1 >=0 ){
          if( y1 < m_image_height){
            f1 = m_image_depth_data[x1+y1*m_image_width];
          }
        }
      }
    }
    if(isnan(f1)){
      f1 = 10.0f;
    }
    res[i]= std::pair<float, int>(f1, i);
  }
  std::sort(res.begin(), res.end(), pairCompare);

  return res[0].first - m_max_index;
}

void OrdinalDepthFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
}

BOOST_CLASS_EXPORT_IMPLEMENT(OrdinalDepthFeature)
