#include "depth_gradient_patch_feature.hh"

using namespace Rdfs;

DepthGradientPatchFeature::DepthGradientPatchFeature(int x1, int y1, int c, int x2, int y2, bool scale_with_depth):
  m_x1(x1), m_y1(y1), m_c(c), m_x2(x2), m_y2(y2), m_scale_with_depth(scale_with_depth){
  if(m_scale_with_depth){
    m_feature_type = 18;
  }else{
    m_feature_type = 17;
  }

}

float DepthGradientPatchFeature::ExtractFeature(const int x, const int y) const{
  int x1, y1, pw, ph;
  float d = 1.0f;
  if(m_scale_with_depth){
    d = 1.0f  / m_image_depth_data[x+y*m_image_width];
    x1 = x + m_x1 * d;
    y1 = y + m_y1 * d;
    pw = m_x2 * d;
   ph = m_y2 * d;
  }else{
    x1 = x + m_x1;
    y1 = y + m_y1;
   pw = m_x2;
   ph = m_y2;
  }

  int xp1 = x1 - pw;
  int xp2 = x1 + pw;
  int yp1 = y1 - ph;
  int yp2 = y1 + ph;

  xp1 = std::max(0, xp1);
  xp1 = std::min(xp1, m_image_width -1);
  yp1 = std::max(0, yp1);
  yp1 = std::min(yp1, m_image_height -1);
  xp2 = std::max(0, xp2);
  xp2 = std::min(xp2, m_image_width -1);
  yp2 = std::max(0, yp2);
  yp2 = std::min(yp2, m_image_height -1);


  float value = m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp2+1 + (yp2+1)*(m_image_width+1)) +m_c];
  value -=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp1 + (yp2+1)*(m_image_width+1)) +m_c];
  value -=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp2+1 + yp1*(m_image_width+1)) +m_c];
  value +=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp1 + yp1*(m_image_width+1)) +m_c];

//   float value = m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp2 + (yp2)*m_image_width) +m_c];
//   if(xp1){
//     value -=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp1-1 + (yp2)*m_image_width) +m_c];
//   }
//   if(yp1){
//     value -=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp2 + (yp1-1)*m_image_width) +m_c];
//     if(xp1){
//       value +=  m_image_depth_gradient_data[m_image_depth_gradient_bins*(xp1-1 + (yp1-1)*m_image_width) +m_c];
//     }
//   }


  //Average over the patch.
  float actual_patch_size = (xp2-xp1 +1) * (yp2- yp1 +1);
  return value / actual_patch_size;
}

void DepthGradientPatchFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
  m_image_depth_gradient_data = data_image.GetGradient3dData().ptr<float>(0);
  m_image_depth_gradient_bins = data_image.GetGradient3dData().channels();
}

BOOST_CLASS_EXPORT_IMPLEMENT(DepthGradientPatchFeature)
