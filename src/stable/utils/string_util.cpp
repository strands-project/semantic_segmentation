// Local includes
#include "string_util.h"

namespace Utils {

void SwapExtension(std::string &str, const std::string old_ext, const std::string new_ext){
  str.replace(str.find(old_ext), old_ext.size(), new_ext);
}

}