#ifndef _UTILS_STATIC_DATASET_HH_
#define _UTILS_STATIC_DATASET_HH_

// STL includes
#include <string>
#include <vector>

//Local includes
#include "data_image.hh"
#include "configuration.hh"
#include "dataset.hh"

namespace Utils {

class StaticDataset : public virtual Dataset{
  public:
    StaticDataset(Configuration conf);
    virtual ~StaticDataset() {};
    int ImageCount() const;
    std::vector<std::string> const& Filenames() const;
    virtual Utils::DataImage GenerateImage(int index) const = 0;

    enum DataType {
      SEQUENCE = 0,
      TRAIN = 1,
      VALID = 2,
      TRAINVALID = 3,
      TEST = 4,
      ALL = 7
    };
    static StaticDataset* LoadDataset(Configuration conf, DataType data_type, int load_flags);
    static StaticDataset* LoadDataset(const std::string& config_filename, DataType data_type, int load_flags);
  protected:
    std::vector<std::string> List(int type);
    std::vector<std::string> ListInDir(std::string dirname, std::string extension = std::string(""));
    std::vector<std::string> LoadFromList(const std::string& image_list_filename);
    virtual void Load(DataType data_type, int load_flags) = 0;

  protected:
    int                             m_image_count;
    std::vector<std::string>        m_image_filenames;

    std::string m_training_split;
    std::string m_validation_split;
    std::string m_test_split;

};
}

#endif // _UTILS__STATIC_DATASET_HH_
