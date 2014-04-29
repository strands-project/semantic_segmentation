//Local includes
#include "tree_feature.hh"


using namespace Rdfs;

void TreeFeature::SetThreshold(const float threshold){
  m_threshold = threshold;
}

float TreeFeature::GetThreshold() const{
  return m_threshold;
}

int TreeFeature::GetFeatureType() const{
  return m_feature_type;
}

bool TreeFeature::Evaluate(const int x, const int y, const Utils::DataImage &data_image){
  return ExtractFeature(x,y, data_image) <= m_threshold;
}

bool TreeFeature::Evaluate(const int x, const int y) const{
  return ExtractFeature(x,y) <= m_threshold;
}

float TreeFeature::ExtractFeature(const int x, const int y, const Utils::DataImage &data_image){
  InitEvaluation(data_image);
  return ExtractFeature(x,y);
}
