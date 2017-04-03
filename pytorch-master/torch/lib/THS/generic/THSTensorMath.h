#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.h"
#else

/* The convention is:
 * spOP has one of the OP as a sparse tensor
 * sspOP returns a sparse result
 * spOPs has all arguments sparse
 *
 * Everything is is up to discretion
 */

TH_API void THSTensor_(zero)(THSTensor *r_);

TH_API void THSTensor_(mul)(THSTensor *r_, THSTensor *t, real value);
TH_API void THSTensor_(div)(THSTensor *r_, THSTensor *t, real value);
TH_API void THSTensor_(cadd)(THSTensor *r_, THSTensor *t, real value, THSTensor *src);
TH_API void THSTensor_(csub)(THSTensor *r_, THSTensor *t, real value, THSTensor *src);
TH_API void THSTensor_(cmul)(THSTensor *r_, THSTensor *t, THSTensor *src);

TH_API void THTensor_(spaddcmul)(THTensor *r_, THTensor *t, real value, THSTensor *src1, THSTensor *src2);

// dense = beta * dense + alpha * sparse * dense
TH_API void THSTensor_(spaddmm)(THTensor *r_, real beta, THTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
// sparse = beta * sparse + alpha * sparse * dense
TH_API void THSTensor_(sspaddmm)(THSTensor *r_, real beta, THSTensor *t, real alpha, THSTensor *sparse, THTensor *dense);
TH_API void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse);

#endif
