/*
 * tistdtypes.h - Compatibility wrapper for AWRL6844
 *
 * TIDL code includes "tistdtypes.h". For AWRL6844, we've already defined
 * the types via preprocessor macros to avoid conflicts.
 */

#ifndef TISTDTYPES_COMPAT_H_
#define TISTDTYPES_COMPAT_H_

#ifdef TARGET_C66X_AWRL6844

/* Types already handled via _TI_STD_TYPES and TISTDTYPES_H_ defines */
/* SDK headers have been blocked from redefining these */

/* Ensure basic types are available */
#include <stdint.h>
#include <stdbool.h>

/* Define types that TIDL code expects if not already defined */
#ifndef _UINT32_T_DECLARED
typedef uint32_t Uint32;
typedef uint16_t Uint16;
typedef uint8_t  Uint8;
typedef int32_t  Int32;
typedef int16_t  Int16;
typedef int8_t   Int8;
#endif

#else

/* PC emulation: use TIDL's local type definitions */
#include "../../dmautils/inc/edma_csl/tistdtypes.h"

#endif

#endif /* TISTDTYPES_COMPAT_H_ */
