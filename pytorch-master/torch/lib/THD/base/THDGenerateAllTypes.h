#ifndef THD_GENERIC_FILE
#error "You must define THD_GENERIC_FILE before including THDGenerateAllTypes.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define THDInf UCHAR_MAX
#define THD_REAL_IS_BYTE
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_BYTE

#define real char
#define accreal long
#define Real Char
#define THDInf CHAR_MAX
#define THD_REAL_IS_CHAR
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define THDInf SHRT_MAX
#define THD_REAL_IS_SHORT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define THDInf INT_MAX
#define THD_REAL_IS_INT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define THDInf LONG_MAX
#define THD_REAL_IS_LONG
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define THDInf FLT_MAX
#define THD_REAL_IS_FLOAT
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THDInf DBL_MAX
#define THD_REAL_IS_DOUBLE
#line 1 THD_GENERIC_FILE
#include THD_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THDInf
#undef THD_REAL_IS_DOUBLE

#undef THD_GENERIC_FILE
