#include "time_stamp.hh"

#include <time.h>
#include <iostream>
#include <stdio.h>

#ifdef WIN32
#include <windows.h>
#endif

using namespace Utils;

TimeStamp::TimeStamp() {
  m_start = 1;
  m_overflow = 0;

  #ifdef WIN32
  m_is_paused = false;
  #endif
}

#ifndef WIN32
void TimeStamp::Stamp() {
  gettimeofday(&m_prev_time, &m_tz);

  if (m_start == 1) {
    m_start = 0;
    m_start_time = m_prev_time;
  }
}
#endif

#ifdef WIN32
void TimeStamp::Stamp() {
  if (m_is_paused) {
    UnPause();
  }

  LARGE_INTEGER time, freq;
  QueryPerformanceCounter(&time);
  QueryPerformanceFrequency(&freq);
  m_prev_time = (double) time.QuadPart / (double) freq.QuadPart;

  if (m_start == 1) {
    m_start = 0;
    m_start_time = m_prev_time;
  }
}
#endif

#ifndef WIN32
// Returns very precise time in seconds since last "stamp"
double TimeStamp::Elapsed() {
  if (m_start == 1) {
    m_start = 0;
    return 0;
  }

  // Get current time
  struct timeval curr_time;
  struct timezone curr_tz;
  gettimeofday(&curr_time, &curr_tz);

  double t1 = (double)m_prev_time.tv_sec + (double) m_prev_time.tv_usec/(1000*1000);
  double t2 = (double)curr_time.tv_sec + (double)curr_time.tv_usec/(1000*1000);
  return t2-t1;
}
#endif

#ifdef WIN32
double TimeStamp::Elapsed() {
  if (m_start == 1) {
    m_start = 0;
    return 0;
  }

  // Get current time
  LARGE_INTEGER time, freq;
  QueryPerformanceCounter(&time);
  QueryPerformanceFrequency(&freq);
  double curr_time = (double)time.QuadPart / (double) freq.QuadPart;

  double elapsed;
  if (m_is_paused) {
    UnPause();
    elapsed = curr_time - m_prev_time;
    Pause();
  }
  else {
    elapsed = curr_time - m_prev_time;
  }

  return elapsed;
}
#endif

#ifndef WIN32
double TimeStamp::TotalElapsed() {
  if (m_start == 1) {
    m_start = 0;
    return 0;
  }

  // Get current time
  struct timeval curr_time;
  struct timezone curr_tz;
  gettimeofday(&curr_time, &curr_tz);

  double t1 = (double)m_start_time.tv_sec + (double) m_start_time.tv_usec/(1000*1000);
  double t2 = (double)curr_time.tv_sec + (double)curr_time.tv_usec/(1000*1000);
  return t2-t1;
}
#endif

#ifdef WIN32
double TimeStamp::TotalElapsed() {
  if (m_start == 1) {
    m_start = 0;
    return 0;
  }

  // Get current time
  LARGE_INTEGER time, freq;
  QueryPerformanceCounter(&time);
  QueryPerformanceFrequency(&freq);
  double curr_time = (double)time.QuadPart / (double) freq.QuadPart;

  return curr_time - m_start_time;
}
#endif

int TimeStamp::ElapsedFrames(double frame_time, double factor) {
  double total =  ((Elapsed() / (frame_time/1000)) + m_overflow) * factor;
  int result = (int) total;
  m_overflow = total - result;

  return result;
}

#ifdef WIN32
// allow timer to be pauses in between "stamps"
void TimeStamp::Pause() {
  if (m_is_paused) {
    return;
  }

  // Get current time
  LARGE_INTEGER time, freq;
  QueryPerformanceCounter(&time);
  QueryPerformanceFrequency(&freq);

  m_pause_time = (double)time.QuadPart / (double) freq.QuadPart;
  m_is_paused = true;
}

// unpause the timer...
void TimeStamp::UnPause() {
  if (!m_is_paused) {
    return;
  }

  // Get current time
  LARGE_INTEGER time, freq;
  QueryPerformanceCounter(&time);
  QueryPerformanceFrequency(&freq);
  double m_curr_time = (double)time.QuadPart / (double) freq.QuadPart;

  m_prev_time += m_curr_time - m_pause_time;
  m_is_paused = false;
}
#endif
