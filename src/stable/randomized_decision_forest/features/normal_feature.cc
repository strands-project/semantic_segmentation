// Local includes
#include "normal_feature.hh"

//Eigen inlcudes
#include "Eigen/LU"

using namespace Rdfs;

NormalFeature::NormalFeature(int x1, int y1, int x2, int y2, int c, Eigen::Vector3f vec): m_x1(x1), m_y1(y1), m_x2(x2), m_y2(y2), m_c(c){
  m_feature_type = NORMAL_FEATURE;
  m_vec_0 = vec(0);
  m_vec_1 = vec(1);
  m_vec_2 = vec(2);
}

float NormalFeature::ExtractFeature(const int x, const int y) const{
//  Eigen::Vector3f normal = m_image_normal_data[x+y*m_image_width];
//  Eigen::Vector3f rectified = m_image_accelerometer_data * normal;
//  return rectified(m_c); //rectified normal value;
  float current = 1.0f/m_image_depth_data[x+y*m_image_width];
  int x1, x2, y1, y2;
  x1 = x+ (static_cast<float>(m_x1) * current +0.5f);
  x2 = x+ (static_cast<float>(m_x2) * current +0.5f);
  y1 = y+ (static_cast<float>(m_y1) * current +0.5f);
  y2 = y+ (static_cast<float>(m_y2) * current +0.5f);

  x1 = std::max(0, x1);
  x1 = std::min(x1, m_image_width -1);
  y1 = std::max(0, y1);
  y1 = std::min(y1, m_image_height -1);
  x2 = std::max(0, x2);
  x2 = std::min(x2, m_image_width -1);
  y2 = std::max(0, y2);
  y2 = std::min(y2, m_image_height -1);
//  std::cout << m_x1 << " " << m_y1 << "     " << m_x2 << "  " << m_y2 << std::endl;

  switch(m_c){
  case 0:{ // Distance
    float f1 = m_image_depth_data[x1+y1*m_image_width];
    float f2 = m_image_depth_data[x2+y2*m_image_width];
    Eigen::Vector3f point1;
    point1 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f1,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f1,
        f1;
    Eigen::Vector3f point2;
    point2 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f2,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f2,
        f2;
    Eigen::Vector3f distance = point1 - point2;
//	 std::cout << "distance " <<  sqrt(pow(distance(0),2.0)+pow(distance(2),2.0)+pow(distance(1),2.0)) << std::endl;
    return sqrt(pow(distance(0),2.0)+pow(distance(2),2.0)+pow(distance(1),2.0));
    break;}
  case 1:{ //Angle between normals
    Eigen::Vector3f normal1;
    normal1 << m_image_normal_data[3*(x1+y1*m_image_width)] , m_image_normal_data[3*(x1+y1*m_image_width)+1] , m_image_normal_data[3*(x1+y1*m_image_width)+2];
    Eigen::Vector3f normal2;
    normal2 << m_image_normal_data[3*(x2+y2*m_image_width)] , m_image_normal_data[3*(x2+y2*m_image_width)+1] , m_image_normal_data[3*(x2+y2*m_image_width)+2];
//	 std::cout << "angle between " << normal1.dot(normal2)<< std::endl;
    return normal1.dot(normal2);
    break;}
  case 2:{ //Angle at p1
    float f1 = m_image_depth_data[x1+y1*m_image_width];
    float f2 = m_image_depth_data[x2+y2*m_image_width];
    Eigen::Vector3f point1;
    point1 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f1,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f1,
        f1;
    Eigen::Vector3f point2;
    point2 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f2,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f2,
        f2;
    Eigen::Vector3f distance = point1 - point2;
    distance.normalize();
    Eigen::Vector3f normal1;
    normal1 << m_image_normal_data[3*(x1+y1*m_image_width)] , m_image_normal_data[3*(x1+y1*m_image_width)+1] , m_image_normal_data[3*(x1+y1*m_image_width)+2];
//	 std::cout <<"angle at p1"  << distance.dot(normal1)<< std::endl;
    return distance.dot(normal1);
    break;}
  case 3:{ //Angle at p1
    float f1 = m_image_depth_data[x1+y1*m_image_width];
    float f2 = m_image_depth_data[x2+y2*m_image_width];
    Eigen::Vector3f point1;
    point1 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f1,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f1,
        f1;
    Eigen::Vector3f point2;
    point2 << (static_cast<float>(x) - m_image_cx) * m_image_fx_inv * f2,
        (static_cast<float>(y) - m_image_cy) * m_image_fy_inv * f2,
        f2;
    Eigen::Vector3f distance = point1 - point2;
    distance.normalize();
    Eigen::Vector3f normal2;
    normal2 << m_image_normal_data[3*(x2+y2*m_image_width)] , m_image_normal_data[3*(x2+y2*m_image_width)+1] , m_image_normal_data[3*(x2+y2*m_image_width)+2];
  //  std::cout <<"angle at p2"  << distance.dot(normal2)<< std::endl;
    return distance.dot(normal2);
    break;}
  case 4:{ //Compare to vector
    Eigen::Vector3f normal;
    normal << m_image_normal_data[3*(x+y*m_image_width)] , m_image_normal_data[3*(x+y*m_image_width)+1] , m_image_normal_data[3*(x+y*m_image_width)+2];
    //std::cout << val << std::endl;
    return m_vector.dot(m_image_accelerometer_data*normal);
    break;}
  default:
    throw std::runtime_error("Shouldn't happen! (In the normal feature)");
    break;
  }

}

void NormalFeature::InitEvaluation(const Utils::DataImage &data_image) {
  m_image_width = data_image.Width();
  m_image_height = data_image.Height();
  m_image_accelerometer_data = data_image.GetAccelerometerData();
  m_image_cx = data_image.m_cx_rgb;
  m_image_cy = data_image.m_cy_rgb;
  m_image_fx_inv = data_image.m_fx_rgb_inv;
  m_image_fy_inv = data_image.m_fy_rgb_inv;
  m_image_accelerometer_data.transposeInPlace();
  m_image_normal_data = data_image.GetNormals().ptr<float>(0);
  m_image_depth_data = data_image.GetDepthImage().ptr<float>(0);
  m_vector = Eigen::Vector3f(m_vec_0, m_vec_1, m_vec_2);
}

BOOST_CLASS_EXPORT_IMPLEMENT(NormalFeature)
