#ifndef _UTILS_TIMESTAMP_H_
#define _UTILS_TIMESTAMP_H_

#include <iostream>

#ifndef WIN32
#include <sys/time.h>
#endif

namespace Utils {
  class TimeStamp {

  public:
    /**
     * @brief Constructor (default)
     */
    TimeStamp();
  
    void Stamp();
    /**
     * @brief Gives target time in seconds since last "stamp"
     */
    double Elapsed();

    /**
     * @brief Gives target total time since first stamp
     */
    double TotalElapsed();

    /**
     * @brief Gives target time in millisecs per frames
     *
     * @param frame_time Time per frame in milliseconds
     * @param factor Scaling factor used to speed and slow the timer
     *
     * @return The # of frames that have elapsed since the last "stamp"
     */
    int ElapsedFrames(double frame_time, double factor=1.0);

#ifdef WIN32
    /**
     * @brief Allow timer to be paused in between "stamps"
     */
    void Pause();

    /**
     * @brief Unpause the the timer...
     */
    void UnPause();
#endif
  
  private:
    int m_start;

#ifndef WIN32
    struct timeval m_prev_time, m_start_time;
    struct timezone m_tz;
#else
    double m_prev_time, m_start_time;
    double m_pause_time;
    bool m_is_paused;
#endif

    double m_overflow;
  };
}

#endif /* _UTILS_TIMESTAMP_H_ */

