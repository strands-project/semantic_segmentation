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
 * @file   Configuration.cc
 * @author Georgios Floros <georgiosfloros@gmail.com>
 * @date   Fri Aug 31 17:53:13 2012
 *
 * @brief  Class for reading named values from configuration files (Implementation)
 *
 * @par    The code has been adapted from the code of Richard Wagner (wagner@umic.edu)
 *
 */
// Local includes
#include "configuration.hh"

// STL-Headers
#include <fstream>
#include <iostream>


using namespace Utils;
using std::string;

Configuration::Configuration(string filename, string delimiter, string comment, string sentry) : m_delimiter(delimiter), m_comment(comment), m_sentry(sentry) {
  // Construct a ConfigFile, getting keys and values from given file
  std::ifstream in(filename.c_str());
  if (!in) {
    throw file_not_found(filename);
  }
  in >> (*this);
}

Configuration::Configuration() : m_delimiter(string(1, '=')), m_comment(string(1, '#')) {
  // Construct a ConfigFile without a file; empty
}

std::vector<int> Configuration::ParseIntVectorFromString(const std::string s){
  std::vector<int> values;
  std::stringstream ss(s);
  while( ss.good() ){
    std::string substr;
    getline( ss, substr, ',' );
    values.push_back( atoi(substr.c_str()) );
  }
  return values;
}

void Configuration::remove(const string &key) {
  // Remove key and its value
  m_contents.erase(m_contents.find(key));
  return;
}

bool Configuration::keyExists(const string &key) const {
  // Indicate whether key is found
  mapci p = m_contents.find(key);
  return (p != m_contents.end());
}

void Configuration::trim(string &s) {
  // Remove leading trailing whitespace
  static const char whitespace[] = " \n\t\v\r\f";
  s.erase(0, s.find_first_not_of(whitespace));
  s.erase(s.find_last_not_of(whitespace) + 1U);
}

namespace Utils {
std::ostream& operator<<(std::ostream& os, const Configuration& cf) {
  // Save a ConfigFile to os
  for (Configuration::mapci p = cf.m_contents.begin();
       p != cf.m_contents.end();++p) {
    os << p->first << " " << cf.m_delimiter << " ";
    os << p->second << std::endl;
  }
  return os;
}

std::istream& operator>>(std::istream& is, Configuration& cf) {
  // Load a ConfigFile from is
  // Read in keys and values, keeping internal whitespace
  typedef string::size_type pos;

  // Separator
  const string& delim = cf.m_delimiter;

  // Comment
  const string& comm = cf.m_comment;

  // End of file sentry
  const string& sentry = cf.m_sentry;

  // Length of separator
  const pos skip = delim.length();

  // Might need to read ahead to see where value ends
  string nextline = "";

  while (is || nextline.length() > 0) {
    // Read an entire line at a time
    string line;
    if (nextline.length() > 0) {
      // We read ahead; Use it now
      line = nextline;
      nextline = "";
    } else {
      std::getline(is, line);
    }

    // Ignore comments
    line = line.substr(0, line.find(comm));

    // Check for end of file sentry
    if (sentry != "" && line.find(sentry) != string::npos) {
      return is;
    }

    // Parse the line if it contains a delimiter
    pos delimPos = line.find(delim);
    if (delimPos < string::npos) {
      // Extract the key
      string key = line.substr(0, delimPos);
      line.replace(0, delimPos + skip, "");

      // See if value continues on the next line
      // Stop at blank line, next line with a key,
      // end of stream, or end of file sentry
      bool terminate = false;
      while (!terminate && is) {
        std::getline(is, nextline);
        terminate = true;

        string nlcopy = nextline;
        Configuration::trim(nlcopy);
        if (nlcopy == "") {
          continue;
        }

        nextline = nextline.substr(0, nextline.find(comm));
        if (nextline.find(delim) != string::npos) {
          continue;
        }

        if (sentry != "" && nextline.find(sentry) != string::npos) {
          continue;
        }

        nlcopy = nextline;
        Configuration::trim(nlcopy);
        if (nlcopy != "") {
          line += "\n";
        }

        line += nextline;
        terminate = false;
      }

      // Store key and value
      Configuration::trim(key);
      Configuration::trim(line);

      // Overwrites if key is repeated
      cf.m_contents[key] = line;
    }
  }
  return is;
}
}
