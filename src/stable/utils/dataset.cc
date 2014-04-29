// Local includes
#include "dataset.hh"

// STL includes
#include <iostream>
#include <stdexcept>

namespace Utils{

bool t_sort(std::string i, std::string j) {
  double i_t = strtod(i.substr(i.find_first_of("-") + 1, i.find_last_of("-") - 2).data(), NULL);
  double j_t = strtod(j.substr(j.find_first_of("-") + 1, j.find_last_of("-") - 2).data(), NULL);
  return i_t < j_t;
}
  
void DisplayLoadFlags(int load_flags) {
 std::cout << std::endl <<  "The following attributes are computed or stored for each image:" << std::endl;

  for ( int flag = Utils::RGB; flag != (Utils::NORMALS << 1); flag = flag << 1 ){
    if(load_flags & flag){
      switch(flag){
      case Utils::RGB :
        std::cout << "RGB Image" << std::endl;
        break;
      case Utils::LAB :
        std::cout << "LAB Image" << std::endl;
        break;
      case Utils::DEPTH :
        std::cout << "Depth Image" << std::endl;
        break;
      case Utils::ANNOTATION :
        std::cout << "Annotation Image" << std::endl;
        break;
      case Utils::UNARY :
        std::cout << "First Unary potential" << std::endl;
        break;
      case Utils::UNARY2 :
        std::cout << "Second Unary potential" << std::endl;
        break;
      case Utils::ACCELEROMETER :
        std::cout << "Accelerometer Data" << std::endl;
        break;
      case Utils::GRADIENT_COLOR :
        std::cout << "Color Gradients" << std::endl;
        break;
      case Utils::GRADIENT_DEPTH:
        std::cout << "Depth Gradients" << std::endl;
        break;
      case Utils::GEOMETRIC_FEAT:
        std::cout << "Geometric features" << std::endl;
        break;
      case Utils::DEPTH_COVARIANCE:
        std::cout << "3D Depth covariance" << std::endl;
        break;
      case Utils::LAB_INTEGRAL:
        std::cout << "Color integral images" << std::endl;
        break;
      case Utils::NORMALS:
        std::cout << "Normal vectors" << std::endl;
        break;
      default:
        throw std::runtime_error("Wrong flag specified during loading of the data!");
        break;
      }
    }
  }
  std::cout << std::endl;
}

const ColorCoding* Dataset::GetColorCoding() const {
  return m_color_coding;
}


}