#include "color_gradient_pixel_feature.hh"

using namespace Rdfs;

ColorGradientPixelFeature::ColorGradientPixelFeature(int x1, int y1, int c1, int x2, int y2, int c2):
  m_x1(x1), m_y1(y1), m_c1(c1), m_x2(x2), m_y2(y2), m_c2(c2){
  m_feature_type = COLOR_GRADIENT;

}

float ColorGradientPixelFeature::ExtractFeature(const int x, const int y) const{
  int x1, y1;
  x1 = x + m_x1;
  y1 = y + m_y1;

  x1 = std::max(0, x1);
  x1 = std::min(x1, m_image_width -1);
  y1 = std::max(0, y1);
  y1 = std::min(y1, m_image_height -1);


  float value1 = m_image_gradient_data[m_image_gradient_bins*(x1+1 + (y1+1)*(m_image_width+1)) +m_c1];
  value1 -=  m_image_gradient_data[m_image_gradient_bins*(x1 + (y1+1)*(m_image_width+1)) +m_c1];
  value1 -=  m_image_gradient_data[m_image_gradient_bins*(x1+1 + y1*(m_image_width+1)) +m_c1];
  value1 +=  m_image_gradient_data[m_image_gradient_bins*(x1 + y1*(m_image_width+1)) +m_c1];

//   float value1 = m_image_gradient_data[m_image_gradient_bins*(x1 + y1*m_image_width) +m_c1];
//   // float value = m_gradient_histogram(xp2, yp2, gradient_band);
// 
//   if(x1 >0){
//     //value -=  m_gradient_histogram(xp1-1, yp2, gradient_band);
//     value1 -=  m_image_gradient_data[m_image_gradient_bins*(x1-1 + y1*m_image_width) +m_c1];
//   }
//   if(y1 >0){
//     //value -=  m_gradient_histogram(xp2, y1-1, gradient_band);
//     value1 -=  m_image_gradient_data[m_image_gradient_bins*(x1 + (y1-1)*m_image_width) +m_c1];
//     if(x1 >0){
//       //value +=  m_gradient_histogram(x1-1, y1-1, gradient_band);
//       value1 +=  m_image_gradient_data[m_image_gradient_bins*(x1-1 + (y1-1)*m_image_width) +m_c1];
//     }
//   }

  int x2, y2;
  x2 = x + m_x2;
  y2 = y + m_y2;

  x2 = std::max(0, x2);
  x2 = std::min(x2, m_image_width -1);
  y2 = std::max(0, y2);
  y2 = std::min(y2, m_image_height -1);


  float value2 = m_image_gradient_data[m_image_gradient_bins*(x2+1 + (y2+1)*(m_image_width+1)) +m_c2];
  value2 -=  m_image_gradient_data[m_image_gradient_bins*(x2 + (y2+1)*(m_image_width+1)) +m_c2];
  value2 -=  m_image_gradient_data[m_image_gradient_bins*(x2+1 + y2*(m_image_width+1)) +m_c2];
  value2 +=  m_image_gradient_data[m_image_gradient_bins*(x2 + y2*(m_image_width+1)) +m_c2];

//   float value2 = m_image_gradient_data[m_image_gradient_bins*((x2+1) +(y2+1)*m_image_width) +m_c2];
//   // float value = m_gradient_histogram(xp2, yp2, gradient_band);
//   //value -=  m_gradient_histogram(xp1-1, yp2, gradient_band);
//   value2 -=  m_image_gradient_data[m_image_gradient_bins*(x2 + (y2+1)*m_image_width) +m_c2];
//   //value -=  m_gradient_histogram(xp2, y1-1, gradient_band);
//   value2 -=  m_image_gradient_data[m_image_gradient_bins*(x2+1 + y2*m_image_width) +m_c2];
//   //value +=  m_gradient_histogram(x1-1, y1-1, gradient_band);
//   value2 +=  m_image_gradient_data[m_image_gradient_bins*(x2 + y2*m_image_width) +m_c2];


  return  value1 - value2;
}

void ColorGradientPixelFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_gradient_data = data_image.GetGradientData().ptr<float>(0);
  m_image_gradient_bins = data_image.GetGradientData().channels();
}

BOOST_CLASS_EXPORT_IMPLEMENT(ColorGradientPixelFeature)
