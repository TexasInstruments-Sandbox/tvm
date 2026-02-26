/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cycle_counter.h
 * \brief 64-bit cycle counter API for DSP benchmarking
 *
 * This header provides a cycle-accurate timing API for TVM DSP runtime.
 * On C66x/C7x DSP targets, it uses the TSCL/TSCH hardware registers.
 * On host emulation, it uses high-resolution timers (chrono/clock_gettime).
 *
 * Features:
 *   - 64-bit cycle count (no overflow concerns for long measurements)
 *   - Automatic measurement overhead calibration
 *   - Simple C API: init/getCount/elapsed
 *   - C++ class wrapper with RAII semantics
 *   - ScopedTimer for easy profiling of code blocks
 *
 * Usage (C-style):
 * \code
 *   TVMDSPCycleCounter_init();
 *   uint64_t start = TVMDSPCycleCounter_getCount64();
 *   // ... code to benchmark ...
 *   uint64_t stop = TVMDSPCycleCounter_getCount64();
 *   uint64_t cycles = TVMDSPCycleCounter_elapsed(start, stop);
 * \endcode
 *
 * Usage (C++ class):
 * \code
 *   tvm::dsp::CycleCounter counter;
 *   counter.start();
 *   // ... code to benchmark ...
 *   counter.stop();
 *   printf("Cycles: %llu\n", (unsigned long long)counter.elapsed());
 * \endcode
 */

#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_CYCLE_COUNTER_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_CYCLE_COUNTER_H_

#include <stdint.h>
#include <stdio.h>

/* Platform-specific includes */
#if defined(TVM_DSP_TARGET_C66X)
#ifdef __TI_COMPILER_VERSION__
#include <c6x.h> /* TSCL, TSCH, _itoll */
#endif
#elif defined(TVM_DSP_TARGET_C7X)
#ifdef __TI_COMPILER_VERSION__
#include <c7x.h> /* TSCL, TSCH, _itoll - C7x also supports these */
#endif
#elif defined(TVM_DSP_TARGET_HOST)
#ifdef __cplusplus
#include <chrono>
#else
#include <time.h>
#endif
#endif

/* ========================================================================== */
/*                          Global Variables                                  */
/* ========================================================================== */

/*! Overhead cycles for calling getCount64 twice (calibrated at init) */
static uint64_t g_tvm_dsp_cycle_overhead = 0;

/* ========================================================================== */
/*                          Function Declarations                             */
/* ========================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Initialize the cycle counter
 *
 * Enables the timestamp counter (on DSP) and calibrates measurement overhead.
 * Must be called once before using other cycle counter functions.
 *
 * \note On C66x, TSCL/TSCH are read-only registers. Writing to TSCL enables
 *       the counter but does not reset it - it free-runs from power-on.
 */
static inline void TVMDSPCycleCounter_init(void) {
#if defined(TVM_DSP_TARGET_C66X)
#ifdef __TI_COMPILER_VERSION__
  uint64_t t_start, t_stop;

  /* Enable timestamp counter by writing to TSCL (value is ignored) */
  TSCL = 0;

  /* Calibrate overhead of reading timestamp twice */
  t_start = _itoll(TSCH, TSCL);
  t_stop = _itoll(TSCH, TSCL);
  g_tvm_dsp_cycle_overhead = t_stop - t_start;
#else
  g_tvm_dsp_cycle_overhead = 0;
#endif
#elif defined(TVM_DSP_TARGET_C7X)
#ifdef __TI_COMPILER_VERSION__
  uint64_t t_start, t_stop;

  /* C7x TSC is always running - cannot reset, just calibrate overhead */
  t_start = __TSC;
  t_stop = __TSC;
  g_tvm_dsp_cycle_overhead = t_stop - t_start;
#else
  g_tvm_dsp_cycle_overhead = 0;
#endif
#else
  /* Host emulation: no overhead to calibrate */
  g_tvm_dsp_cycle_overhead = 0;
#endif
}

/*!
 * \brief Get current 64-bit cycle count
 *
 * Returns the current timestamp counter value. On DSP targets, this is
 * the hardware cycle counter. On host, it's nanoseconds since epoch
 * (simulating 1 cycle = 1 nanosecond at 1 GHz).
 *
 * \return Current cycle count as 64-bit value
 */
static inline uint64_t TVMDSPCycleCounter_getCount64(void) {
#if defined(TVM_DSP_TARGET_C66X)
#ifdef __TI_COMPILER_VERSION__
  /* Use _itoll intrinsic for atomic 64-bit read of TSCH:TSCL */
  return _itoll(TSCH, TSCL);
#else
  return 0;
#endif
#elif defined(TVM_DSP_TARGET_C7X)
#ifdef __TI_COMPILER_VERSION__
  /* C7x: Use __TSC intrinsic for 64-bit timestamp counter */
  return __TSC;
#else
  return 0;
#endif
#elif defined(TVM_DSP_TARGET_HOST)
#ifdef __cplusplus
  /* C++: use chrono high-resolution clock */
  using namespace std::chrono;
  auto now = high_resolution_clock::now();
  auto ns = duration_cast<nanoseconds>(now.time_since_epoch()).count();
  return (uint64_t)ns;
#else
  /* C: use clock_gettime */
  struct timespec ts;
#if defined(__APPLE__) || (defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0)
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#else
  return 0;
#endif
#endif
#else
  return 0;
#endif
}

/*!
 * \brief Get low 32-bit cycle count only
 *
 * Returns just the lower 32 bits of the cycle count for compatibility
 * with 32-bit APIs. Note: overflows in ~8 seconds at 500 MHz.
 *
 * \return Lower 32 bits of cycle count
 */
static inline uint32_t TVMDSPCycleCounter_getCount32(void) {
#if defined(TVM_DSP_TARGET_C66X)
#ifdef __TI_COMPILER_VERSION__
  return TSCL;
#else
  return 0;
#endif
#elif defined(TVM_DSP_TARGET_C7X)
#ifdef __TI_COMPILER_VERSION__
  /* C7x: Get lower 32 bits of __TSC */
  return (uint32_t)(__TSC & 0xFFFFFFFF);
#else
  return 0;
#endif
#else
  return (uint32_t)(TVMDSPCycleCounter_getCount64() & 0xFFFFFFFF);
#endif
}

/*!
 * \brief Calculate elapsed cycles with overhead compensation
 *
 * Computes cycles between start and stop timestamps, subtracting the
 * calibrated measurement overhead.
 *
 * \param start Cycle count at start of measurement
 * \param stop Cycle count at end of measurement
 * \return Elapsed cycles (overhead-compensated)
 */
static inline uint64_t TVMDSPCycleCounter_elapsed(uint64_t start, uint64_t stop) {
  uint64_t raw = stop - start;
  /* Avoid underflow if measured time is less than overhead */
  return (raw > g_tvm_dsp_cycle_overhead) ? (raw - g_tvm_dsp_cycle_overhead) : 0;
}

/*!
 * \brief Get the calibrated measurement overhead
 *
 * Returns overhead cycles determined during init. Useful for diagnostics.
 *
 * \return Overhead cycles for one start/stop measurement pair
 */
static inline uint64_t TVMDSPCycleCounter_getOverhead(void) { return g_tvm_dsp_cycle_overhead; }

#ifdef __cplusplus
}
#endif

/* ========================================================================== */
/*                          C++ Class Interface                               */
/* ========================================================================== */

#ifdef __cplusplus

namespace tvm {
namespace dsp {

/*!
 * \class CycleCounter
 * \brief RAII-style cycle counter for DSP benchmarking
 *
 * Provides a convenient object-oriented interface for cycle-accurate
 * timing measurements. Automatically initializes on first use.
 */
class CycleCounter {
 public:
  /*!
   * \brief Construct cycle counter
   * \param autoInit If true (default), initialize hardware counter
   */
  explicit CycleCounter(bool autoInit = true) : startCount_(0), stopCount_(0), running_(false) {
    if (autoInit) {
      initOnce();
    }
  }

  /*! \brief Start timing measurement */
  void start() {
    startCount_ = TVMDSPCycleCounter_getCount64();
    running_ = true;
  }

  /*! \brief Stop timing measurement */
  void stop() {
    stopCount_ = TVMDSPCycleCounter_getCount64();
    running_ = false;
  }

  /*!
   * \brief Get elapsed cycles (overhead-compensated)
   * \return Elapsed cycles between start() and stop()
   */
  uint64_t elapsed() const { return TVMDSPCycleCounter_elapsed(startCount_, stopCount_); }

  /*!
   * \brief Get raw elapsed cycles (no overhead compensation)
   * \return Raw elapsed cycles
   */
  uint64_t elapsedRaw() const { return stopCount_ - startCount_; }

  /*!
   * \brief Get current elapsed without stopping
   * \return Current elapsed cycles since start()
   */
  uint64_t currentElapsed() const {
    uint64_t now = TVMDSPCycleCounter_getCount64();
    return TVMDSPCycleCounter_elapsed(startCount_, now);
  }

  /*!
   * \brief Check if timer is running
   * \return true if start() called but stop() not yet called
   */
  bool isRunning() const { return running_; }

  /*!
   * \brief Get measurement overhead
   * \return Calibrated overhead cycles
   */
  static uint64_t getOverhead() { return TVMDSPCycleCounter_getOverhead(); }

 private:
  uint64_t startCount_;
  uint64_t stopCount_;
  bool running_;

  static void initOnce() {
    static bool initialized = false;
    if (!initialized) {
      TVMDSPCycleCounter_init();
      initialized = true;
    }
  }
};

/*!
 * \class ScopedTimer
 * \brief RAII timer that prints elapsed cycles on destruction
 *
 * Usage:
 * \code
 *   {
 *     tvm::dsp::ScopedTimer timer("MyFunction");
 *     // ... code to time ...
 *   }  // Prints "MyFunction: 12345 cycles" on scope exit
 * \endcode
 */
class ScopedTimer {
 public:
  /*!
   * \brief Construct and start scoped timer
   * \param name Label to print with timing result (can be nullptr)
   */
  explicit ScopedTimer(const char* name = nullptr) : name_(name) { counter_.start(); }

  /*! \brief Destructor prints elapsed cycles */
  ~ScopedTimer() {
    counter_.stop();
    if (name_) {
      printf("%s: %llu cycles\n", name_, (unsigned long long)counter_.elapsed());
    } else {
      printf("Elapsed: %llu cycles\n", (unsigned long long)counter_.elapsed());
    }
  }

  /*!
   * \brief Get current elapsed without stopping
   * \return Current elapsed cycles
   */
  uint64_t currentElapsed() const { return counter_.currentElapsed(); }

 private:
  CycleCounter counter_;
  const char* name_;

  /* Non-copyable */
  ScopedTimer(const ScopedTimer&);
  ScopedTimer& operator=(const ScopedTimer&);
};

}  // namespace dsp
}  // namespace tvm

#endif /* __cplusplus */

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_CYCLE_COUNTER_H_ */
