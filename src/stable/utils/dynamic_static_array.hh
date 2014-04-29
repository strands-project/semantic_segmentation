#ifndef _UTILS_DYNAMICSTATICARRAY_H_
#define _UTILS_DYNAMICSTATICARRAY_H_

#include <assert.h>

namespace Utils {
  template <typename T>
  struct DynamicStaticArray {
    int S;
    T *data;
    DynamicStaticArray(): S(0), data(0) {
    }

    DynamicStaticArray(int size): S(0), data(0) {
      InitSize(size);
    }

    ~DynamicStaticArray() {
      if (S > 0) {
	delete [] data;
      }
    }

    __inline T& operator[](const int i) {
      return data[i];
    }

    const __inline T& operator[](const int i) const {
      return data[i];
    }

    void Fill(const T &t) {
      for (int i = 0; i < S; i++) {
	data[i] = t;
      }
    }

    void InitSize(const int Snew) {
      assert(Snew >= 0);
      if (S > 0) {
	delete [] data;
      }
      S = Snew;
      data = new T[S];
    }

    const T& operator()(const int x, const int y, const int w) const {
      return (*this)[y*w+x];
    }

    T& operator()(const int x, const int y, const int w) {
      return (*this)[y*w+x];
    }

  private:
    DynamicStaticArray(const DynamicStaticArray&);
    DynamicStaticArray& operator=(const DynamicStaticArray&);
  };
}

#endif /* _UTILS_DYNAMICSTATICARRAY_H_ */
