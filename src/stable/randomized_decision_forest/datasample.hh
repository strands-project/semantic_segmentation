#ifndef DATASAMPLE_HH
#define DATASAMPLE_HH

namespace Rdfs {

class DataSample {
public:
  unsigned int image_id;
  unsigned short x_pos;
  unsigned short y_pos;
  signed char    label;
  double         label_weight;
  signed char    center_label;
  // double         histogram_weight;
  //unsigned int   sample_id;

public:
  DataSample(unsigned int image_id=0, unsigned int x_pos=0, unsigned int y_pos=0,
             signed char label=0, double label_weight=0, signed char center_label=0){
    this->image_id = image_id;
    this->x_pos = x_pos;
    this->y_pos = y_pos;
    this->label = label;
    this->label_weight = label_weight;
    this->center_label = center_label;
  }

  static bool DataSampleSort(DataSample a, DataSample b){
    return a.image_id < b.image_id;
  }
  static bool DataSamplePointerSort(DataSample *a, DataSample *b){
    return a->image_id < b->image_id;
  }
};
}
#endif // DATASAMPLE_HH
