#include "cluster3d.hh"

using namespace Inference;

int Cluster3D::m_window;
int Cluster3D::m_class_count;


void Cluster3D::Init(int reconstruction_window, int class_count){
  m_window = reconstruction_window;
  m_class_count = class_count;
}

Cluster3D::Cluster3D(int frame_index){
  m_last_fused_frame = frame_index;
  m_class_distribution = std::vector<float>(m_class_count,1.0f/static_cast<float>(m_class_count));
  m_class_distribution_accu = std::vector<float>(m_class_count,0);
  m_use_dense_results=false;
  m_accu_is_empty=true;
  m_class_distribution_accu_sum_counter=0;
//  m_fuse_counter =1;
}

void Cluster3D::PushBackFusedPoint(int frame_index){
  m_last_fused_frame = frame_index;
//  m_fuse_counter++;
}

void Cluster3D::SetPosition(float x, float y, float z){
  m_x3 = x;
  m_y3 = y;
  m_z3 = z;
}

void Cluster3D::GetPosition(float *x3, float *y3, float *z3) const{
  (*x3)= m_x3;
  (*y3)= m_y3;
  (*z3)= m_z3;
}

void Cluster3D::SetColor(unsigned char r, unsigned char g, unsigned char b){
  m_color_r = r;
  m_color_g = g;
  m_color_b = b;
}

void Cluster3D::GetColor(unsigned char *r, unsigned char *g, unsigned char *b) const{
  (*r) = m_color_r;// / m_fuse_counter;
  (*g) = m_color_g;// / m_fuse_counter;
  (*b) = m_color_b;// / m_fuse_counter;
}

void Cluster3D::SetNormal(float normal_x3, float normal_y3, float normal_z3){
  m_normal_x3 = normal_x3;
  m_normal_y3 = normal_y3;
  m_normal_z3 = normal_z3;
}

void Cluster3D::GetNormal(float *normal_x3, float *normal_y3, float *normal_z3) const{
  (*normal_x3)= m_normal_x3;// / m_fuse_counter;
  (*normal_y3)= m_normal_y3;// / m_fuse_counter;
  (*normal_z3)= m_normal_z3;// / m_fuse_counter;
}


void Cluster3D::SetPointDistribution(std::vector<float> &dist){
  m_use_dense_results =true;
  m_accu_is_empty=true;
  for(int c=0; c < m_class_count; c++){
    m_class_distribution[c] = dist[c];
    m_class_distribution_accu[c]=0;
  }
  m_class_distribution_accu_sum_counter=0;


}

std::vector<float> const& Cluster3D::GetDistribution() const{
  if(m_use_dense_results){
    return m_class_distribution;
  }else{
    return m_class_distribution_accu;
  }
}

float Cluster3D::GetAccuDistributionSum() const{
  return m_class_distribution_accu_sum_counter;
}

bool Cluster3D::IsAccuEmpty() const{
  return m_accu_is_empty;
}

int Cluster3D::GetLastFrameIndex() const{
  return m_last_fused_frame;
}

void Cluster3D::AddDistributionToAccu(std::vector<float> &dist, float weight){
  m_accu_is_empty=false;
  for(int c=0; c < m_class_count; c++){
    m_class_distribution_accu[c] += weight*dist[c];
  }
  m_class_distribution_accu_sum_counter+=weight;
}

const std::vector<float> &Cluster3D::GetAccuDistribution() const{
  return m_class_distribution_accu;
}






