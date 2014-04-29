#ifndef _INFERENCE_ENERGY_HH_
#define _INFERENCE_ENERGY_HH_

namespace Inference {
  template <class T>
  class Energy {
  public:
    T *m_unary_cost;
    T *m_pairwise_cost;
    T *m_higher_cost;
    T *m_higher_truncation;
    T *m_car_cost;
    T *m_car_truncation;
    int m_num_labels;
    int m_num_nodes;
    int m_num_edges;
    int m_num_higher;
    int m_num_cars;
    int *m_pair_index;
    int **m_higher_index;
    int **m_car_index;
    int *m_higher_elements;
    int *m_car_elements;
    float **m_car_weight;

    /**
   * @brief Constrcutor
   *
   * @param num_labels Number of labels
   * @param num_nodes Number of nodes of the graph
   * @param num_edges Number of edges of the graph
   * @param num_higher Number of higher order nodes
   * @param num_cars Number of car detection nodes
   */
    Energy(int num_labels, int num_nodes, int num_edges, int num_higher, int num_cars);

    /**
   * @brief Destructor
   *
   */
    ~Energy();

    /**
   * @brief Allocate higher order indices
   *
   */
    void AllocateHigherIndexes();
  };

  template <class T>
  Energy<T>::Energy(int num_labels, int num_nodes, int num_edges, int num_higher, int num_cars) {
    m_num_labels = num_labels;
    m_num_nodes = num_nodes;
    m_num_edges = num_edges;
    m_num_higher = num_higher;
    m_num_cars = num_cars;

    m_unary_cost = new T[m_num_nodes * m_num_labels];
    m_pair_index = new int[m_num_edges * 2];
    m_pairwise_cost = new T[m_num_edges];

    m_higher_cost = new T[m_num_higher * (m_num_labels + 1)];
    m_car_cost = new T[m_num_cars * (m_num_labels + 1)];
    m_higher_elements = new int[m_num_higher];
    m_car_elements = new int[m_num_cars];
    m_higher_truncation = new T[m_num_higher];
    m_car_truncation = new T[m_num_cars];
    m_higher_index = new int *[m_num_higher];
    m_car_index = new int *[m_num_cars];
    m_car_weight = new float *[m_num_cars];
    memset(m_higher_index, 0, m_num_higher * sizeof(int *));
    memset(m_car_index, 0, m_num_cars * sizeof(int *));
    memset(m_car_weight, 0, m_num_cars * sizeof(float *));
  }

  template <class T>
  Energy<T>::~Energy() {
    if (m_higher_index != NULL) {
      for (int i = 0; i < m_num_higher; i++) {
        if (m_higher_index[i] != NULL) {
          delete [] m_higher_index[i];
        }
      }
    }
    if (m_car_index != NULL) {
      for (int i = 0; i < m_num_cars; i++) {
        if (m_car_index[i] != NULL) {
          delete [] m_car_index[i];
        }
      }
    }
    if (m_car_weight != NULL) {
      for (int i = 0; i < m_num_cars; i++) {
        if (m_car_weight[i] != NULL) {
          delete [] m_car_weight[i];
        }
      }
    }

    if (m_pairwise_cost != NULL) {
      delete [] m_pairwise_cost;
    }
    if (m_unary_cost != NULL) {
      delete [] m_unary_cost;
    }
    if (m_pair_index != NULL) {
      delete [] m_pair_index;
    }

    if (m_higher_truncation != NULL) {
      delete [] m_higher_truncation;
    }
    if (m_car_truncation != NULL) {
      delete [] m_car_truncation;
    }

    if (m_higher_cost != NULL) {
      delete [] m_higher_cost;
    }
    if (m_car_cost != NULL) {
      delete [] m_car_cost;
    }

    if (m_higher_index != NULL) {
      delete [] m_higher_index;
    }
    if (m_car_index != NULL) {
      delete [] m_car_index;
    }

    if (m_car_weight != NULL) {
      delete [] m_car_weight;
    }

    if (m_higher_elements != NULL) {
      delete [] m_higher_elements;
    }
    if (m_car_elements != NULL) {
      delete [] m_car_elements;
    }
  }

  template <class T>
  void Energy<T>::AllocateHigherIndexes() {
    for (int i = 0; i < m_num_higher; i++) {
      m_higher_index[i] = new int[m_higher_elements[i]];
    }

    for (int i = 0; i < m_num_cars; i++) {
      m_car_index[i] = new int[m_car_elements[i]];
    }
    for (int i = 0; i < m_num_cars; i++) {
      m_car_weight[i] = new float[m_car_elements[i]];
    }
  }
}

#endif // _INFERENCE_ENERGY_HH_
