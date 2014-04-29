#include "geometrical_feature.hh"

using namespace Rdfs;

GeometricalFeature::GeometricalFeature(int c): m_c(c){
  m_feature_type = GEOMETRICAL;
}

float GeometricalFeature::ExtractFeature(const int x, const int y) const{
  return   m_image_geometrical_data[m_image_geometrical_feature_count*(x+y*m_image_width) + m_c];

}

void GeometricalFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_geometrical_data = data_image.Get3DFeature().ptr<float>(0);
  m_image_geometrical_feature_count = data_image.Get3DFeature().channels();
}

BOOST_CLASS_EXPORT_IMPLEMENT(GeometricalFeature)
