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
 * \file cpp/scope_guard.h
 * \brief RAII scope guard for automatic resource cleanup
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * This header provides a "scope guard" - a mechanism to ensure cleanup code
 * runs when leaving a scope, regardless of HOW you leave (normal return,
 * early return on error, etc).
 *
 * PROBLEM IT SOLVES:
 *
 * In C code, you often write patterns like this:
 *
 *   void* buffer = allocate();
 *   if (error1) {
 *     free(buffer);     // Must remember to free!
 *     return -1;
 *   }
 *   if (error2) {
 *     free(buffer);     // Must remember again!
 *     return -1;
 *   }
 *   // ... more code ...
 *   free(buffer);       // And at the end
 *   return 0;
 *
 * This is error-prone: if you add a new error path, you might forget to free.
 *
 * WITH SCOPE GUARD:
 *
 *   void* buffer = allocate();
 *   TVM_DSP_SCOPE_EXIT(free(buffer));  // Cleanup registered once
 *
 *   if (error1) {
 *     return -1;  // buffer automatically freed!
 *   }
 *   if (error2) {
 *     return -1;  // buffer automatically freed!
 *   }
 *   return 0;      // buffer automatically freed!
 *
 * =============================================================================
 * C++ CONCEPTS USED (for C programmers)
 * =============================================================================
 *
 * 1. TEMPLATES (template <typename F>)
 *    - Like a "code generator" that creates specialized code for each type
 *    - ScopeGuard<F> means "ScopeGuard that stores something of type F"
 *    - When you write MakeScopeGuard(some_lambda), the compiler figures out
 *      what F is and generates code for that specific type
 *
 * 2. RAII (Resource Acquisition Is Initialization)
 *    - C++ objects have constructors (called when created) and destructors
 *      (called automatically when they go out of scope)
 *    - We use the destructor to run cleanup code
 *    - This is automatic - you don't call the destructor manually
 *
 * 3. LAMBDAS ([&]() { code; })
 *    - An inline anonymous function
 *    - [&] means "capture all local variables by reference" so we can use them
 *    - () is the parameter list (empty)
 *    - { code; } is the function body
 *    - Example: [&]() { free(buffer); } creates a function that frees buffer
 *
 * 4. MOVE SEMANTICS (std::move)
 *    - Transfers ownership of resources without copying
 *    - After std::move(x), x is in a "moved-from" state (don't use it)
 *    - More efficient than copying for complex objects
 *    - For our lambda, it just transfers the function object efficiently
 *
 * 5. NOEXCEPT
 *    - A promise that a function won't throw exceptions
 *    - Required for efficient move operations
 *    - Our DSP code doesn't use exceptions, so this is always safe
 *
 * 6. DELETED FUNCTIONS (= delete)
 *    - Explicitly disables a function
 *    - ScopeGuard(const ScopeGuard&) = delete means "cannot copy"
 *    - This prevents accidentally having two guards that both try to cleanup
 *
 * =============================================================================
 * MEMORY MODEL
 * =============================================================================
 *
 * NO DYNAMIC ALLOCATION. All data is stored inline:
 * - The cleanup function (lambda) is stored directly in the guard object
 * - The active flag is a single bool
 * - Everything lives on the stack where the guard is declared
 *
 * =============================================================================
 * USAGE EXAMPLES
 * =============================================================================
 *
 * Basic usage with the macro:
 *
 *   void* ptr = tvm_dsp_alloc(size, align, pool);
 *   if (!ptr) return -1;
 *   TVM_DSP_SCOPE_EXIT(tvm_dsp_free(ptr));  // Will free when scope exits
 *
 *   // ... use ptr safely ...
 *   // ptr is freed automatically at end of function
 *
 * Multiple guards:
 *
 *   void* buf1 = allocate();
 *   TVM_DSP_SCOPE_EXIT(free(buf1));
 *
 *   void* buf2 = allocate();
 *   TVM_DSP_SCOPE_EXIT(free(buf2));
 *
 *   // Both freed in reverse order (buf2 first, then buf1)
 *
 * Dismissing the guard (when transferring ownership):
 *
 *   void* buffer = allocate();
 *   auto guard = tvm::dsp::MakeScopeGuard([&]() { free(buffer); });
 *
 *   if (success) {
 *     guard.Dismiss();  // Don't free - caller takes ownership
 *     return buffer;
 *   }
 *   return nullptr;  // Error path: buffer freed automatically
 *
 * =============================================================================
 */

#ifndef TVM_DSP_RUNTIME_CPP_SCOPE_GUARD_H_
#define TVM_DSP_RUNTIME_CPP_SCOPE_GUARD_H_

#include <utility>  /* std::move - for transferring ownership efficiently */

namespace tvm {
namespace dsp {

/*!
 * \brief RAII scope guard that executes cleanup on destruction
 *
 * This class stores a "cleanup function" (typically a lambda) and calls it
 * when the ScopeGuard object is destroyed (goes out of scope).
 *
 * \tparam F The type of the cleanup function. Usually deduced automatically
 *           when using MakeScopeGuard() or TVM_DSP_SCOPE_EXIT().
 *
 * HOW IT WORKS:
 * 1. Constructor stores the cleanup function and sets active_ = true
 * 2. When the object goes out of scope, C++ automatically calls ~ScopeGuard()
 * 3. The destructor checks active_ and calls cleanup_() if true
 *
 * WHY TEMPLATE?
 * Each lambda has a unique type (even if the code looks identical).
 * Using a template lets us store ANY callable type without heap allocation.
 * The compiler generates a specialized ScopeGuard class for each lambda type.
 */
template <typename F>
class ScopeGuard {
 public:
  /*!
   * \brief Construct a scope guard with the given cleanup function
   *
   * \param cleanup The function to call when this guard is destroyed.
   *                Usually a lambda like [&]() { free(ptr); }
   *
   * The cleanup function is MOVED into the guard (not copied) for efficiency.
   * After construction, the guard is "active" and will run cleanup on
   * destruction.
   */
  explicit ScopeGuard(F cleanup)
      : cleanup_(std::move(cleanup)), active_(true) {}

  /*!
   * \brief Destructor - executes cleanup if still active
   *
   * This is called automatically by C++ when the guard goes out of scope.
   * If the guard is still active (not dismissed), it calls the cleanup
   * function.
   *
   * IMPORTANT: This is where the "magic" happens. You never call this
   * manually - C++ calls it for you when the variable goes out of scope.
   */
  ~ScopeGuard() {
    if (active_) {
      cleanup_();
    }
  }

  /*!
   * \brief Move constructor - transfers ownership from another guard
   *
   * \param other The guard to move from. After the move, 'other' is inactive.
   *
   * This allows guards to be returned from functions or stored in containers.
   * The source guard is deactivated to prevent double-cleanup.
   *
   * noexcept tells the compiler this won't throw exceptions, enabling
   * optimizations.
   */
  ScopeGuard(ScopeGuard&& other) noexcept
      : cleanup_(std::move(other.cleanup_)), active_(other.active_) {
    other.active_ = false;  /* Prevent other from running cleanup */
  }

  /*!
   * \brief Dismiss the scope guard, preventing cleanup execution
   *
   * Call this when:
   * - The resource has been successfully transferred to someone else
   * - You've handled cleanup manually for some reason
   * - The success path shouldn't free the resource
   *
   * After calling Dismiss(), the destructor will NOT run the cleanup function.
   *
   * Example:
   *   void* ptr = allocate();
   *   auto guard = MakeScopeGuard([&]() { free(ptr); });
   *
   *   if (success) {
   *     guard.Dismiss();  // Caller takes ownership, don't free
   *     return ptr;
   *   }
   *   return nullptr;  // Failed: guard will free ptr
   */
  void Dismiss() { active_ = false; }

  /*!
   * \brief Check if the guard is still active
   * \return true if cleanup will execute on destruction, false if dismissed
   *
   * Mostly useful for testing/debugging.
   */
  bool IsActive() const { return active_; }

  /*
   * DELETED FUNCTIONS
   *
   * These lines DISABLE certain operations that would be dangerous:
   *
   * - Copy constructor: Can't copy a guard because then BOTH would try
   *   to run cleanup, causing double-free bugs.
   *
   * - Copy assignment: Same reason as copy constructor.
   *
   * - Move assignment: Disabled for simplicity. Guards should be created
   *   once and not reassigned.
   *
   * If you try to copy a ScopeGuard, you'll get a compile error.
   */
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

 private:
  F cleanup_;      /*!< The cleanup function to execute */
  bool active_;    /*!< Whether cleanup should execute on destruction */
};

/*!
 * \brief Create a scope guard with automatic type deduction
 *
 * \tparam F Callable type (automatically deduced by the compiler)
 * \param cleanup The cleanup function (usually a lambda)
 * \return A ScopeGuard instance that will call cleanup on destruction
 *
 * WHY THIS FUNCTION EXISTS:
 *
 * In C++11/14, you can't write:
 *   ScopeGuard guard([&]() { ... });  // Error: must specify template type
 *
 * You'd have to write:
 *   ScopeGuard<decltype(lambda)> guard(lambda);  // Ugly!
 *
 * This helper function uses "template argument deduction" - the compiler
 * figures out F from the argument you pass. So you can write:
 *   auto guard = MakeScopeGuard([&]() { ... });  // Nice!
 *
 * The TVM_DSP_SCOPE_EXIT macro uses this internally.
 *
 * Example:
 *   auto guard = MakeScopeGuard([&]() {
 *     printf("Cleaning up!\n");
 *     free(buffer);
 *   });
 */
template <typename F>
ScopeGuard<F> MakeScopeGuard(F cleanup) {
  return ScopeGuard<F>(std::move(cleanup));
}

}  // namespace dsp
}  // namespace tvm

/*
 * =============================================================================
 * MACRO IMPLEMENTATION
 * =============================================================================
 *
 * The macro creates a uniquely-named variable to hold the guard.
 * Using __LINE__ ensures each TVM_DSP_SCOPE_EXIT on a different line
 * gets a different variable name, avoiding conflicts.
 */

/* Helper: Concatenate two tokens. Two levels needed due to macro expansion. */
#define TVM_DSP_SCOPE_GUARD_CONCAT_IMPL(a, b) a##b
#define TVM_DSP_SCOPE_GUARD_CONCAT(a, b) TVM_DSP_SCOPE_GUARD_CONCAT_IMPL(a, b)

/*!
 * \brief Create an anonymous scope guard that executes code on scope exit
 *
 * \param code The code to execute when the current scope exits
 *
 * HOW IT WORKS:
 * 1. Creates a lambda [&]() { code; } that captures all locals by reference
 * 2. Passes it to MakeScopeGuard() to create a guard
 * 3. Stores the guard in a uniquely-named variable (using __LINE__)
 * 4. When the scope exits, the guard's destructor runs, which calls the lambda
 *
 * The [&] capture means the lambda can access and modify any local variable
 * that exists where the macro is used. This is why free(ptr) works even
 * though ptr is defined outside the lambda.
 *
 * Example:
 *   void* ptr = allocate();
 *   TVM_DSP_SCOPE_EXIT(free(ptr));
 *   // Expands to something like:
 *   // auto tvm_dsp_scope_guard_42 = MakeScopeGuard([&]() { free(ptr); });
 *
 * LIMITATIONS:
 * - Can only have one TVM_DSP_SCOPE_EXIT per line (due to __LINE__)
 * - The code runs in reverse order of declaration (last guard first)
 */
#define TVM_DSP_SCOPE_EXIT(code)                                             \
  auto TVM_DSP_SCOPE_GUARD_CONCAT(tvm_dsp_scope_guard_, __LINE__) =           \
      ::tvm::dsp::MakeScopeGuard([&]() { code; })

#endif  /* TVM_DSP_RUNTIME_CPP_SCOPE_GUARD_H_ */
