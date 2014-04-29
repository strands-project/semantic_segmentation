#ifndef CLUSTER_3D_H
#define CLUSTER_3D_H

#include <deque>
#include <vector>
#include <iostream>

namespace Inference{

class Cluster3D {

private:

  float m_x3;
  float m_y3;
  float m_z3;

  float m_normal_x3;
  float m_normal_y3;
  float m_normal_z3;

  unsigned char m_color_r;
  unsigned char m_color_g;
  unsigned char m_color_b;


  static int m_window;
  static int m_class_count;
  int m_last_fused_frame;
  int m_fuse_counter;

  std::vector<float> m_class_distribution;
  std::vector<float> m_class_distribution_accu;
  float m_class_distribution_accu_sum_counter;
  bool m_use_dense_results;
  bool m_accu_is_empty;



public:
  static void Init(int reconstruction_window, int class_count);

  Cluster3D(int frame_index);

  void PushBackFusedPoint(int frame_index);

  void SetPosition(float x,float y,float z);

  void GetPosition(float *x3, float *y3, float*z3) const;

  void SetColor(unsigned char r, unsigned char g, unsigned char b);

  void GetColor(unsigned char *r, unsigned char *g, unsigned char *b) const;

  void SetNormal(float normal_x3, float normal_y3, float normal_z3);

  void GetNormal(float *normal_x3, float *normal_y3, float *normal_z3) const;

  int GetLastFrameIndex() const;


  void AddDistributionToAccu(std::vector<float> &dist, float weight = 1.0f);
  std::vector<float> const& GetAccuDistribution() const;
  void SetPointDistribution(std::vector<float> &dist);


  std::vector<float> const& GetDistribution() const;
  float GetAccuDistributionSum() const;

  bool IsAccuEmpty() const;
};
}

#endif 

