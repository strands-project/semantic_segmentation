// Copyright (c) 2004 Richard J. Wagner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

/**
 * \file   Configuration.hh
 * \author Georgios Floros <georgiosfloros@gmail.com>
 * \date   Fri Aug 31 17:51:39 2012
 * 
 * \brief  Class for reading named values from configuration files
 * 
 * \par    The code has been adapted from the code of Richard Wagner (wagner@umic.edu)
 * 
 */

#ifndef _UTILS_CONFIGURATION_H_
#define _UTILS_CONFIGURATION_H_

// STL-Headers
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <iostream>

namespace Utils {
  class Configuration {
  protected:
    // Separator between key and value
    std::string m_delimiter;

    // Separator between value and comments
    std::string m_comment;

    // Optional string to signal end of file
    std::string m_sentry;

    // Extracted keys and values
    std::map<std::string, std::string> m_contents;

    typedef std::map<std::string, std::string>::iterator mapi;
    typedef std::map<std::string, std::string>::const_iterator mapci;

  public:
    Configuration(std::string filename, std::string delimiter = "=", 
		  std::string comment = "#", std::string sentry = "EndConfigFile");

    Configuration();

    // Search for key and read value or optional default value
    template<class T> T read(const std::string &key) const;
    template<class T> T read(const std::string &key, const T& value) const;
    template<class T> bool readInto(T& var, const std::string& key) const;
    template<class T> bool readInto(T& var, const std::string& key, const T& value) const;


    static std::vector<int> ParseIntVectorFromString(const std::string s);

    // Modify keys and values
    template<class T> void add(std::string key, const T& value);
    void remove(const std::string& key);

    // Check whether key exists in configuration
    bool keyExists(const std::string& key) const;

    // Check or change configuration syntax
    std::string delimiter() const {
      return m_delimiter;
    }

    std::string comment() const {
      return m_comment;
    }

    std::string sentry() const {
      return m_sentry;
    }

    std::string set_delimiter(const std::string& delimiter) {
      std::string old = m_delimiter;
      m_delimiter = delimiter;
      return old;
    }

    std::string set_comment(const std::string& comment) {
      std::string old = m_comment;
      m_comment = comment;
      return old;
    }

    // Write or read configuration
    friend std::ostream& operator<<(std::ostream& os, const Configuration& cf);
    friend std::istream& operator>>(std::istream& is, Configuration& cf);

  protected:
    template<class T> static std::string T_as_string(const T& t);
    template<class T> static T string_as_T(const std::string& s);
    static void trim(std::string& s);

    // Exception types
  public:
    struct file_not_found {
      std::string filename;
      file_not_found(const std::string& filename_ = std::string()) : filename(filename_) { }
    };

    struct key_not_found {
      // Thrown only by T read(key) variant of read()
      std::string key;
      key_not_found(const std::string& key_ = std::string()) : key(key_) { }
    };
  };

  template<class T>
  std::string Configuration::T_as_string(const T& t) {
    // Convert from a T to a string
    // Type T must be support << operator
    std::ostringstream ost;
    ost << t;
    return ost.str();
  }

  template<class T>
  T Configuration::string_as_T(const std::string& s) {
    // Convert from a string to a T
    // Type T must support >> operator
    T t;
    std::istringstream ist(s);
    ist >> t;
    return t;
  }

  // Static
  template<>
  inline std::string Configuration::string_as_T<std::string>(const std::string& s) {
    // Convert from a string to a string
    // In other words do nothing
    return s;
  }

  // Static
  template<>
  inline bool Configuration::string_as_T<bool>(const std::string& s) {
    // Convert from a string to a bool
    // Interpret "false", "F", "no", "0" as false
    // Interpret "true", "T", "yes", "y", "1", "-1", or anything else as true
    bool b = true;
    std::string sup = s;
    for (std::string::iterator p = sup.begin();p != sup.end(); ++p) {
      *p = (char)toupper(*p);
    }
    if (sup == std::string("FALSE") || sup == std::string("F") || sup == std::string("NO") ||  sup == std::string("N") ||  sup == std::string("0") || sup == std::string("NONE")) {
      b = false;
    }
    return b;
  }

  template<class T>
  T Configuration::read(const std::string& key) const {
    // Read the value corresponding to key
    mapci p = m_contents.find(key);
    if (p == m_contents.end()) {
      std::cerr << key << std::endl;
      throw key_not_found(key);
    }
    return string_as_T<T>(p->second);
  }
  
  template<class T>
  T Configuration::read(const std::string& key, const T& value) const {
    // Return the value corresponding to key or
    // given default value if key is not found
    mapci p = m_contents.find(key);
    if (p == m_contents.end()) {
      return value;
    }
    return string_as_T<T>(p->second);
  }

  template<class T>
  bool Configuration::readInto(T& var, const std::string& key) const {
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise leave var untouched
    mapci p = m_contents.find(key);
    bool found = (p != m_contents.end());
    if (found) {
      var = string_as_T<T>(p->second);
    }
    return found;
  }

  template<class T>
  bool Configuration::readInto(T& var, const std::string& key, const T& value) const {
    // Get the value corresponding to key and store in var
    // Return true if key is found
    // Otherwise set var to given default
    mapci p = m_contents.find(key);
    bool found = (p != m_contents.end());
    if (found) {
      var = string_as_T<T>(p->second);
    } else {
      var = value;
    }
    return found;
  }

  template<class T>
  void Configuration::add(std::string key, const T& value) {
    // Add a key given value
    std::string v = T_as_string(value);
    trim(key);
    trim(v);
    m_contents[key] = v;
    return;
  }
}

#endif /* _UTILS_CONFIGURATION_H_ */
