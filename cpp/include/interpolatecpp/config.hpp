#pragma once

#include <interpolatecpp/version.hpp>

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef INTERPOLATECPP_EXPORTS
        #define INTERPOLATECPP_API __declspec(dllexport)
    #else
        #define INTERPOLATECPP_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define INTERPOLATECPP_API __attribute__((visibility("default")))
#else
    #define INTERPOLATECPP_API
#endif
