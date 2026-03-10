/*
 * rproc_trace.h - Debug tracing via remoteproc trace buffer
 *
 * Provides RPROC_TRACE_MSG() and RPROC_TRACE_PRINTF() macros that
 * write to the remoteproc trace buffer (DebugP_log) and the shared
 * memory printf buffer respectively.
 *
 * Enable at compile time with -DRPROC_TRACE=1.  When disabled (the
 * default), the macros compile to nothing.
 *
 * The trace buffer is readable from Linux even after a DSP crash:
 *   cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
 *
 * Usage:
 *   RPROC_TRACE_MSG("checkpoint reached");
 *   RPROC_TRACE_PRINTF("value = %d\n", x);
 */

#ifndef RPROC_TRACE_H
#define RPROC_TRACE_H

#ifndef RPROC_TRACE
#define RPROC_TRACE 0
#endif

#if RPROC_TRACE

/*
 * dsp_trace_msg() writes a fixed string to the remoteproc trace
 * buffer via DebugP_log().  It is provided by the firmware and
 * exported to DLOAD modules via the symbol table.  The trace buffer
 * is readable from Linux even after a DSP crash.
 */
#define RPROC_TRACE_MSG(msg) do { \
    extern void dsp_trace_msg(const char *); \
    dsp_trace_msg(msg); \
} while (0)

#else  /* !RPROC_TRACE */

#define RPROC_TRACE_MSG(msg)    ((void)0)

#endif /* RPROC_TRACE */

#endif /* RPROC_TRACE_H */
