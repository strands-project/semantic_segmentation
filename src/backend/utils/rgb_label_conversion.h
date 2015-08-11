#pragma once

// STL includes
#include <map>
#include <string>
#include <type_traits>

// OpenCV includes
#include <opencv2/core/core.hpp>

// Local includes
#include "json/json.h"

typedef int label_type;

namespace Utils{

class RgbLabelConversion{
public:
  RgbLabelConversion(){}

  RgbLabelConversion(const std::string& json_string){
    Json::Reader reader;
    Json::Value mapping;
    reader.parse(json_string, mapping);

    std::string name;
    label_type label;
    uchar r, g, b;
    for(auto c : mapping){
      name = c["name"].asString();
      label = c["label"].asInt();
      r = static_cast<uchar>(c["color"][0].asUInt());
      g = static_cast<uchar>(c["color"][1].asUInt());
      b = static_cast<uchar>(c["color"][2].asUInt());
      _name_to_label[name] = label;
      _label_to_name[label] = name;
      _label_to_r[label] = r;
      _label_to_g[label] = g;
      _label_to_b[label] = b;
      _rgb_to_label[1000000*r+1000*g+b] = label;
    }
    _label_names.reserve(_label_to_name.size());
    for(uint l = 0; l < _label_to_name.size() -1 ; l++){ // -1 to account for the the "void".
      _label_names.push_back(_label_to_name[l]);
    }
  }

  cv::Mat labelToRgb(const cv::Mat& l) const{
    cv::Mat result(l.rows, l.cols, CV_8UC3);
    for(int y = 0; y < l.rows; ++y){
      uchar* c = result.ptr<uchar>(y);
      const label_type* r =l.ptr<label_type>(y);
      for(int x = 0; x < l.cols; ++x){
        labelToRgb(*r, c[2], c[1], c[0]);
        c+=3;
        r++;
      }
    }
    return result;
  }

  cv::Mat rgbToLabel(const cv::Mat& l) const{
    cv::Mat result;
    if(std::is_same<label_type, int>::value){
      result = cv::Mat(l.rows, l.cols, CV_32SC1);
    }else if(std::is_same<label_type, char>::value){
      result = cv::Mat(l.rows, l.cols, CV_8SC1);
    }else if(std::is_same<label_type, short>::value){
      result = cv::Mat(l.rows, l.cols, CV_16SC1);
    }else{
      //Should never happen!
    }
    for(int y = 0; y < l.rows; ++y){
      const uchar* c = l.ptr<uchar>(y);
      label_type* r = result.ptr<label_type>(y);
      for(int x = 0; x < l.cols; ++x){
        rgbToLabel(c[2], c[1], c[0], *r);
        c+=3;
        r++;
      }
    }
    return result;
  }


  void labelToRgb(label_type label, uchar& r, uchar& g, uchar& b) const{
    r = _label_to_r.at(label);
    g = _label_to_g.at(label);
    b = _label_to_b.at(label);
  }

  void rgbToLabel(uchar r, uchar g, uchar b, label_type& label) const{
    label = _rgb_to_label.at(1000000*r+1000*g+b);
  }


  std::string getLabelName(label_type label) const{
    return _label_to_name.at(label);
  }

  label_type getLabelNumber(std::string  label) const{
    return _name_to_label.at(label);
  }

  int getLabeLCount() const{
    return _name_to_label.size();
  }

  int getValidLabelCount() const{
    int i = 0;
    for(auto l : _label_to_name){
      if(l.first >= 0)
        i++;
    }
    return i;
  }

std::vector<std::string> getLabelNames() const{
  return _label_names;
}

private:
  std::map<std::string, label_type> _name_to_label;
  std::map<label_type, std::string> _label_to_name;
  std::map<label_type, uchar> _label_to_r;
  std::map<label_type, uchar> _label_to_g;
  std::map<label_type, uchar> _label_to_b;
  std::map<int, label_type> _rgb_to_label;
  std::vector<std::string> _label_names;
};
}