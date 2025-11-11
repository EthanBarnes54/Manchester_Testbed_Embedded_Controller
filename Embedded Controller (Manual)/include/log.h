#pragma once
#include <Arduino.h>

#ifndef LOG_LEVEL
#define LOG_LEVEL 0 
#endif

#if LOG_LEVEL >= 1 
    #define LOG_ERROR(x) Serial.println(String("[Error has occured:] ") + x)
#else 
    #define LOG_ERROR(x)
#endif

#if LOG_LEVEL >= 2
  #define LOG_INFO(x)    Serial.println(String("[Information is available:] ") + x)
#else
  #define LOG_INFO(x)
#endif

#if LOG_LEVEL >= 3
  #define LOG_DEBUG(x)   Serial.println(String("[Debugging is required:] ") + x)
#else
  #define LOG_DEBUG(x)
#endif

#if LOG_LEVEL >= 4
  #define LOG_TRACE(x)   Serial.println(String("[Trace back is needed:] ") + x)
#else
  #define LOG_TRACE(x)
#endif

