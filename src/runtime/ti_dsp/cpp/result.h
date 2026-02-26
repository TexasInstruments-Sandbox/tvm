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
 * \file cpp/result.h
 * \brief Result type for error handling without exceptions
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * Result<T, E> is a type that represents either a successful value (T) or an
 * error (E). It's similar to Rust's Result type or C++23's std::expected, but
 * works with C++14 and has no dependencies.
 *
 * PROBLEM IT SOLVES:
 *
 * In C, error handling is typically done with return codes:
 *
 *   int parse_config(const char* path, Config* out_config);
 *
 *   Config config;
 *   int err = parse_config("config.txt", &config);
 *   if (err != 0) {
 *     // Handle error... but what was the error?
 *     // Need to look up error code meanings
 *   }
 *
 * Problems with this approach:
 * - Easy to forget to check the return value
 * - Output parameters are awkward
 * - Error codes are just integers - no context
 * - Can't return errors from constructors
 *
 * WITH RESULT:
 *
 *   Result<Config, ParseError> parse_config(const char* path);
 *
 *   auto result = parse_config("config.txt");
 *   if (result.IsOk()) {
 *     Config config = result.Value();
 *     // Use config...
 *   } else {
 *     ParseError err = result.Error();
 *     printf("Parse failed: %s\n", err.message);
 *   }
 *
 * Benefits:
 * - Can't accidentally ignore the result (it holds the value)
 * - Error type is explicit and can carry information
 * - No output parameters needed
 * - Works with any type, not just integers
 *
 * =============================================================================
 * C++ CONCEPTS USED (for C programmers)
 * =============================================================================
 *
 * 1. UNION-LIKE STORAGE
 *    - Result stores EITHER a value OR an error, never both
 *    - Uses a union internally to share memory between them
 *    - A boolean flag tracks which one is currently stored
 *
 * 2. STATIC_ASSERT WITH TYPE TRAITS
 *    - We use is_trivially_copyable to ensure types can be safely copied
 *    - Trivially copyable types can be copied with memcpy
 *    - This includes: int, float, pointers, simple structs
 *
 * 3. EXPLICIT CONSTRUCTOR
 *    - The 'explicit' keyword prevents accidental conversions
 *    - You must explicitly create Ok() or Err() results
 *    - This makes the code's intent clear
 *
 * 4. FACTORY FUNCTIONS (Ok, Err)
 *    - Instead of constructors, we use named functions
 *    - Ok(value) creates a success result
 *    - Err(error) creates an error result
 *    - This is clearer than constructor overloads
 *
 * 5. VALUE CATEGORIES (const T& vs T)
 *    - Value() returns a reference to avoid copying
 *    - ValueOr() returns by value because it might return the default
 *
 * =============================================================================
 * MEMORY MODEL
 * =============================================================================
 *
 * NO DYNAMIC ALLOCATION. Result uses a union for storage:
 *
 *   Result<int, ErrorCode> r = Ok(42);
 *
 *   Memory layout:
 *   +----------------------------------+
 *   | value_ OR error_ | is_ok_ (bool) |
 *   +----------------------------------+
 *   |<-- union ------->|
 *
 *   Total size: max(sizeof(T), sizeof(E)) + sizeof(bool) + padding
 *
 * Since we only support trivially copyable types:
 * - No destructor calls needed
 * - Safe to copy with memcpy
 * - No complex lifetime management
 *
 * =============================================================================
 * USAGE EXAMPLES
 * =============================================================================
 *
 * Defining error types:
 *
 *   // Simple error code
 *   enum class ParseError {
 *     kInvalidFormat,
 *     kFileNotFound,
 *     kOutOfMemory
 *   };
 *
 *   // Error with message (must be trivially copyable!)
 *   struct Error {
 *     int code;
 *     const char* message;  // Pointer is trivially copyable
 *   };
 *
 * Returning results:
 *
 *   Result<int, ParseError> parse_int(const char* str) {
 *     if (str == nullptr) {
 *       return Err(ParseError::kInvalidFormat);
 *     }
 *     int value = atoi(str);
 *     return Ok(value);
 *   }
 *
 * Checking and accessing:
 *
 *   auto result = parse_int("42");
 *
 *   // Method 1: Check then access
 *   if (result.IsOk()) {
 *     int value = result.Value();
 *     printf("Got: %d\n", value);
 *   }
 *
 *   // Method 2: Use ValueOr with default
 *   int value = result.ValueOr(-1);  // -1 if error
 *
 *   // Method 3: Check for error
 *   if (result.IsErr()) {
 *     ParseError err = result.Error();
 *     // Handle error...
 *   }
 *
 * Chaining operations (manual):
 *
 *   Result<int, Error> step1();
 *   Result<float, Error> step2(int input);
 *
 *   Result<float, Error> do_both() {
 *     auto r1 = step1();
 *     if (r1.IsErr()) {
 *       return Err(r1.Error());  // Propagate error
 *     }
 *     return step2(r1.Value());
 *   }
 *
 * =============================================================================
 * COMPARISON: Result vs Exceptions vs Error Codes
 * =============================================================================
 *
 * | Feature           | Error Codes | Exceptions  | Result<T,E>      |
 * |-------------------|-------------|-------------|------------------|
 * | Can be ignored    | Yes (bad!)  | No          | Harder to ignore |
 * | Heap allocation   | No          | Often       | No               |
 * | Error context     | Limited     | Rich        | Customizable     |
 * | DSP compatible    | Yes         | No          | Yes              |
 * | Type safety       | Weak        | Strong      | Strong           |
 * | Performance       | Best        | Worst       | Good             |
 *
 * =============================================================================
 * VOID RESULT (Result<void, E>)
 * =============================================================================
 *
 * For functions that don't return a value but can fail:
 *
 *   Result<void, Error> initialize() {
 *     if (failed) {
 *       return Err(Error{-1, "Init failed"});
 *     }
 *     return Ok();  // Success with no value
 *   }
 *
 *   auto result = initialize();
 *   if (result.IsErr()) {
 *     // Handle error
 *   }
 *
 * =============================================================================
 */

#ifndef TVM_DSP_RUNTIME_CPP_RESULT_H_
#define TVM_DSP_RUNTIME_CPP_RESULT_H_

#include <cstddef>     /* size_t */
#include <type_traits> /* std::is_trivially_copyable, std::enable_if */

namespace tvm {
namespace dsp {

/*
 * =============================================================================
 * FORWARD DECLARATIONS
 * =============================================================================
 */

template <typename T, typename E>
class Result;

/*
 * =============================================================================
 * TAG TYPES
 * =============================================================================
 * These are empty types used to distinguish Ok and Err construction.
 * They have no runtime cost - just help the compiler understand intent.
 */

/*!
 * \brief Tag type for successful result construction
 *
 * Used internally by the Ok() factory function.
 * You don't need to use this directly.
 */
struct OkTag {};

/*!
 * \brief Tag type for error result construction
 *
 * Used internally by the Err() factory function.
 * You don't need to use this directly.
 */
struct ErrTag {};

/*
 * =============================================================================
 * OK AND ERR WRAPPER TYPES
 * =============================================================================
 * These wrap values to enable type deduction in factory functions.
 */

/*!
 * \brief Wrapper for a success value
 *
 * Created by the Ok() factory function. Implicitly converts to Result.
 *
 * \tparam T The value type (always a non-reference type)
 */
template <typename T>
struct OkValue {
  T value;

  explicit OkValue(const T& v) : value(v) {}
};

/*!
 * \brief Wrapper for an error value
 *
 * Created by the Err() factory function. Implicitly converts to Result.
 *
 * \tparam E The error type (always a non-reference type)
 */
template <typename E>
struct ErrValue {
  E error;

  explicit ErrValue(const E& e) : error(e) {}
};

/*!
 * \brief Wrapper for void success (no value)
 *
 * Created by Ok() with no arguments for Result<void, E>.
 */
struct OkVoid {};

/*
 * =============================================================================
 * FACTORY FUNCTIONS
 * =============================================================================
 * Use these to create Result values. They're clearer than constructors.
 */

/*!
 * \brief Create a success result with a value
 *
 * \tparam T The value type (automatically deduced, references stripped)
 * \param value The success value
 * \return OkValue<T> that converts to Result<T, E>
 *
 * Example:
 *   Result<int, Error> r = Ok(42);
 *
 * NOTE: We use typename std::decay<T>::type to strip references and cv-qualifiers.
 * This ensures that Ok(lvalue) creates OkValue<T>, not OkValue<T&>.
 */
template <typename T>
OkValue<typename std::decay<T>::type> Ok(T&& value) {
  return OkValue<typename std::decay<T>::type>(value);
}

/*!
 * \brief Create a success result with no value (for void results)
 *
 * \return OkVoid that converts to Result<void, E>
 *
 * Example:
 *   Result<void, Error> r = Ok();
 */
inline OkVoid Ok() { return OkVoid{}; }

/*!
 * \brief Create an error result
 *
 * \tparam E The error type (automatically deduced, references stripped)
 * \param error The error value
 * \return ErrValue<E> that converts to Result<T, E>
 *
 * Example:
 *   Result<int, Error> r = Err(Error{-1, "failed"});
 *
 * NOTE: We use typename std::decay<E>::type to strip references and cv-qualifiers.
 * This ensures that Err(lvalue) creates ErrValue<E>, not ErrValue<E&>.
 */
template <typename E>
ErrValue<typename std::decay<E>::type> Err(E&& error) {
  return ErrValue<typename std::decay<E>::type>(error);
}

/*
 * =============================================================================
 * RESULT CLASS (Main Implementation)
 * =============================================================================
 */

/*!
 * \brief A type that holds either a success value or an error
 *
 * Result<T, E> represents the outcome of an operation that can fail.
 * It contains either:
 * - A value of type T (success case)
 * - An error of type E (failure case)
 *
 * \tparam T The success value type. Must be trivially copyable.
 * \tparam E The error type. Must be trivially copyable.
 *
 * RESTRICTIONS:
 * Both T and E must be trivially copyable (int, float, pointers, simple
 * structs). This ensures no dynamic allocation and simple copying.
 *
 * THREAD SAFETY:
 * Result is not thread-safe. If multiple threads access the same Result,
 * you must provide external synchronization.
 */
template <typename T, typename E>
class Result {
  static_assert(std::is_trivially_copyable<T>::value,
                "Result value type T must be trivially copyable");
  static_assert(std::is_trivially_copyable<E>::value,
                "Result error type E must be trivially copyable");

 public:
  using ValueType = T;
  using ErrorType = E;

  /*
   * =========================================================================
   * CONSTRUCTORS
   * =========================================================================
   */

  /*!
   * \brief Construct a success result from OkValue
   *
   * \param ok The wrapped success value (from Ok() factory)
   *
   * This allows: Result<int, Error> r = Ok(42);
   */
  Result(const OkValue<T>& ok) : is_ok_(true) { storage_.value = ok.value; }

  /*!
   * \brief Construct an error result from ErrValue
   *
   * \param err The wrapped error value (from Err() factory)
   *
   * This allows: Result<int, Error> r = Err(error);
   */
  Result(const ErrValue<E>& err) : is_ok_(false) { storage_.error = err.error; }

  /*!
   * \brief Copy constructor
   */
  Result(const Result& other) : is_ok_(other.is_ok_) {
    if (is_ok_) {
      storage_.value = other.storage_.value;
    } else {
      storage_.error = other.storage_.error;
    }
  }

  /*!
   * \brief Copy assignment
   */
  Result& operator=(const Result& other) {
    if (this != &other) {
      is_ok_ = other.is_ok_;
      if (is_ok_) {
        storage_.value = other.storage_.value;
      } else {
        storage_.error = other.storage_.error;
      }
    }
    return *this;
  }

  /*
   * =========================================================================
   * STATE CHECKING
   * =========================================================================
   */

  /*!
   * \brief Check if this is a success result
   * \return true if this contains a value, false if it contains an error
   *
   * Example:
   *   if (result.IsOk()) {
   *     auto value = result.Value();
   *   }
   */
  bool IsOk() const { return is_ok_; }

  /*!
   * \brief Check if this is an error result
   * \return true if this contains an error, false if it contains a value
   *
   * Example:
   *   if (result.IsErr()) {
   *     auto error = result.Error();
   *   }
   */
  bool IsErr() const { return !is_ok_; }

  /*!
   * \brief Boolean conversion - true if success
   *
   * Allows using Result in if statements:
   *   if (result) { ... }  // Same as if (result.IsOk())
   */
  explicit operator bool() const { return is_ok_; }

  /*
   * =========================================================================
   * VALUE ACCESS
   * =========================================================================
   */

  /*!
   * \brief Get the success value
   * \return Reference to the contained value
   *
   * WARNING: Calling this when IsOk() is false is undefined behavior!
   * Always check IsOk() first.
   *
   * Example:
   *   if (result.IsOk()) {
   *     int value = result.Value();
   *   }
   */
  T& Value() { return storage_.value; }

  /*!
   * \brief Get the success value (const version)
   */
  const T& Value() const { return storage_.value; }

  /*!
   * \brief Get the value or a default if this is an error
   *
   * \param default_value The value to return if this is an error
   * \return The contained value if success, default_value if error
   *
   * This is safe to call without checking IsOk() first.
   *
   * Example:
   *   int value = result.ValueOr(-1);  // -1 if error
   */
  T ValueOr(const T& default_value) const {
    return is_ok_ ? storage_.value : default_value;
  }

  /*!
   * \brief Get the error value
   * \return Reference to the contained error
   *
   * WARNING: Calling this when IsErr() is false is undefined behavior!
   * Always check IsErr() first.
   *
   * Example:
   *   if (result.IsErr()) {
   *     Error err = result.Error();
   *   }
   */
  E& Error() { return storage_.error; }

  /*!
   * \brief Get the error value (const version)
   */
  const E& Error() const { return storage_.error; }

  /*!
   * \brief Get the error or a default if this is success
   *
   * \param default_error The error to return if this is success
   * \return The contained error if failure, default_error if success
   */
  E ErrorOr(const E& default_error) const {
    return is_ok_ ? default_error : storage_.error;
  }

 private:
  /*!
   * \brief Union storage - holds either value or error, never both
   *
   * Since both T and E are trivially copyable, we don't need to
   * worry about constructors or destructors.
   */
  union Storage {
    T value;
    E error;

    /* Default constructor - leaves storage uninitialized */
    Storage() {}
  };

  Storage storage_;  /*!< The value or error */
  bool is_ok_;       /*!< true if storage contains value, false if error */
};

/*
 * =============================================================================
 * RESULT<void, E> SPECIALIZATION
 * =============================================================================
 * For functions that can fail but don't return a value on success.
 */

/*!
 * \brief Specialization for void value type
 *
 * Used for operations that can fail but don't return a value:
 *
 *   Result<void, Error> initialize();
 *
 *   auto result = initialize();
 *   if (result.IsErr()) {
 *     // Handle error
 *   }
 */
template <typename E>
class Result<void, E> {
  static_assert(std::is_trivially_copyable<E>::value,
                "Result error type E must be trivially copyable");

 public:
  using ValueType = void;
  using ErrorType = E;

  /*!
   * \brief Construct a success result (no value)
   *
   * This allows: Result<void, Error> r = Ok();
   */
  Result(OkVoid) : is_ok_(true) {}

  /*!
   * \brief Construct an error result
   *
   * This allows: Result<void, Error> r = Err(error);
   */
  Result(const ErrValue<E>& err) : is_ok_(false), error_(err.error) {}

  /*!
   * \brief Copy constructor
   */
  Result(const Result& other) : is_ok_(other.is_ok_), error_(other.error_) {}

  /*!
   * \brief Copy assignment
   */
  Result& operator=(const Result& other) {
    if (this != &other) {
      is_ok_ = other.is_ok_;
      error_ = other.error_;
    }
    return *this;
  }

  /*
   * =========================================================================
   * STATE CHECKING
   * =========================================================================
   */

  bool IsOk() const { return is_ok_; }
  bool IsErr() const { return !is_ok_; }
  explicit operator bool() const { return is_ok_; }

  /*
   * =========================================================================
   * ERROR ACCESS
   * =========================================================================
   */

  /*!
   * \brief Get the error value
   *
   * WARNING: Calling this when IsErr() is false is undefined behavior!
   */
  E& Error() { return error_; }
  const E& Error() const { return error_; }

  E ErrorOr(const E& default_error) const {
    return is_ok_ ? default_error : error_;
  }

 private:
  E error_;    /*!< The error (only valid if !is_ok_) */
  bool is_ok_; /*!< true if success, false if error */
};

/*
 * =============================================================================
 * COMMON ERROR TYPES
 * =============================================================================
 * Pre-defined error types for common use cases.
 */

/*!
 * \brief Simple error code enumeration
 *
 * A basic set of error codes for common failure scenarios.
 * Use this when you don't need detailed error information.
 */
enum class ErrorCode : int {
  kSuccess = 0,       /*!< No error (shouldn't appear in Result::Error) */
  kInvalidArgument,   /*!< Invalid function argument */
  kOutOfMemory,       /*!< Memory allocation failed */
  kOutOfRange,        /*!< Index or value out of valid range */
  kNotFound,          /*!< Requested item not found */
  kAlreadyExists,     /*!< Item already exists */
  kPermissionDenied,  /*!< Operation not permitted */
  kNotSupported,      /*!< Feature not supported */
  kInternal,          /*!< Internal error */
  kUnknown            /*!< Unknown error */
};

/*!
 * \brief Error with code and message
 *
 * A more informative error type that includes a human-readable message.
 * The message must be a string literal or have static lifetime.
 *
 * Example:
 *   return Err(Error{ErrorCode::kInvalidArgument, "size must be positive"});
 */
struct Error {
  ErrorCode code;       /*!< The error code */
  const char* message;  /*!< Human-readable message (must have static lifetime) */
};

/*!
 * \brief Convenience type alias for Result with Error
 *
 * Example:
 *   ErrorResult<int> parse_int(const char* str);
 */
template <typename T>
using ErrorResult = Result<T, Error>;

/*!
 * \brief Convenience type alias for Result with ErrorCode
 *
 * Example:
 *   CodeResult<int> parse_int(const char* str);
 */
template <typename T>
using CodeResult = Result<T, ErrorCode>;

}  // namespace dsp
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_RESULT_H_ */
