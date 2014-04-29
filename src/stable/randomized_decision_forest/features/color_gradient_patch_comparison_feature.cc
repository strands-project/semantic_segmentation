#include "color_gradient_patch_comparison_feature.hh"

using namespace Rdfs;

ColorGradientPatchComparisonFeature::ColorGradientPatchComparisonFeature(int x1, int y1, int c1, int x2, int y2, int x3, int y3, int c2, int x4, int y4, bool scale_with_depth):
  m_x1(x1), m_y1(y1), m_c1(c1), m_x2(x2), m_y2(y2), m_x3(x3), m_y3(y3), m_c2(c2), m_x4(x4), m_y4(y4), m_scale_with_depth(scale_with_depth)
{
  if(!m_scale_with_depth){
    m_feature_type = COLOR_GRADIENT_PATCH_COMPARISON;
  }else{
    m_feature_type = COLOR_GRADIENT_PATCH_COMPARISON_SCALED;
  }
}

float ColorGradientPatchComparisonFeature::ExtractFeature(const int x, const int y) const{
  int x1, y1, pw, ph;
  float d = 1.0f  / m_image_depth_data[x+y*m_image_width];;
  if(m_scale_with_depth){
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



  float value1 = m_image_gradient_data[m_image_gradient_bins*(xp2+1 + (yp2+1)*(m_image_width+1)) +m_c2];
  value1 -=  m_image_gradient_data[m_image_gradient_bins*(xp1 + (yp2+1)*(m_image_width+1)) +m_c2];
  value1 -=  m_image_gradient_data[m_image_gradient_bins*(xp2+1 + yp1*(m_image_width+1)) +m_c2];
  value1 +=  m_image_gradient_data[m_image_gradient_bins*(xp1 + yp1*(m_image_width+1)) +m_c2];

//  float value1 = m_image_gradient_data[m_image_gradient_bins*(xp2 + yp2*m_image_width) +m_c1];
//  // float value = m_gradient_histogram(xp2, yp2, gradient_band);
//
//   if(xp1 >0){
//     //value -=  m_gradient_histogram(xp1-1, yp2, gradient_band);
//     value1 -=  m_image_gradient_data[m_image_gradient_bins*(xp1-1 + yp2*m_image_width) +m_c1];
//   }
//   if(yp1 >0){
//     //value -=  m_gradient_histogram(xp2, y1-1, gradient_band);
//     value1 -=  m_image_gradient_data[m_image_gradient_bins*(xp2 + (yp1-1)*m_image_width) +m_c1];
//     if(xp1 >0){
//       //value +=  m_gradient_histogram(x1-1, y1-1, gradient_band);
//       value1 +=  m_image_gradient_data[m_image_gradient_bins*(xp1-1 + (yp1-1)*m_image_width) +m_c1];
//     }
//   }

  //Average over the patch.
  float actual_patch_size = (xp2-xp1 +1) * (yp2- yp1 +1);
  value1 = value1 / actual_patch_size;



  if(m_scale_with_depth){
    x1 = x + m_x3 * d;
    y1 = y + m_y3 * d;
    pw = m_x4 * d;
    ph = m_y4 * d;
  }else{
    x1 = x + m_x3;
    y1 = y + m_y3;
    pw = m_x4;
    ph = m_y4;
  }

  xp1 = x1 - pw;
  xp2 = x1 + pw;
  yp1 = y1 - ph;
  yp2 = y1 + ph;

  xp1 = std::max(0, xp1);
  xp1 = std::min(xp1, m_image_width -1);
  yp1 = std::max(0, yp1);
  yp1 = std::min(yp1, m_image_height -1);
  xp2 = std::max(0, xp2);
  xp2 = std::min(xp2, m_image_width -1);
  yp2 = std::max(0, yp2);
  yp2 = std::min(yp2, m_image_height -1);


  float value2 = m_image_gradient_data[m_image_gradient_bins*(xp2+1 + (yp2+1)*(m_image_width+1)) +m_c2];
  value2 -=  m_image_gradient_data[m_image_gradient_bins*(xp1 + (yp2+1)*(m_image_width+1)) +m_c2];
  value2 -=  m_image_gradient_data[m_image_gradient_bins*(xp2+1 + yp1*(m_image_width+1)) +m_c2];
  value2 +=  m_image_gradient_data[m_image_gradient_bins*(xp1 + yp1*(m_image_width+1)) +m_c2];

//   float value2 = m_image_gradient_data[m_image_gradient_bins*(xp2+1 + (yp2+1)*m_image_width) +m_c2];
//   // float value = m_gradient_histogram(xp2, yp2, gradient_band);
//   //value -=  m_gradient_histogram(xp1-1, yp2, gradient_band);
//   value2 -=  m_image_gradient_data[m_image_gradient_bins*(xp1 + (yp2+1)*m_image_width) +m_c2];
//   //value -=  m_gradient_histogram(xp2, y1-1, gradient_band);
//   value2 -=  m_image_gradient_data[m_image_gradient_bins*(xp2+1 + yp1*m_image_width) +m_c2];
//   //value +=  m_gradient_histogram(x1-1, y1-1, gradient_band);
//   value2 +=  m_image_gradient_data[m_image_gradient_bins*(xp1 + yp1*m_image_width) +m_c2];



  //Average over the patch.
  actual_patch_size = (xp2-xp1 +1) * (yp2- yp1 +1);
  value2 = value2 / actual_patch_size;

  return  value1 - value2;
}

void ColorGradientPatchComparisonFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
  m_image_gradient_data = data_image.GetGradientData().ptr<float>(0);
  m_image_gradient_bins = data_image.GetGradientData().channels();
}

BOOST_CLASS_EXPORT_IMPLEMENT(ColorGradientPatchComparisonFeature)
