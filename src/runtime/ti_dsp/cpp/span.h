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
 * \file cpp/span.h
 * \brief Non-owning view over contiguous memory (like std::span from C++20)
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * Span is a lightweight, non-owning view over a contiguous sequence of elements.
 * It's similar to std::span (C++20) but works with C++14 and has no dependencies.
 *
 * PROBLEM IT SOLVES:
 *
 * In C code, arrays are often passed as pointer + size pairs:
 *
 *   void process_shape(const int64_t* shape, size_t ndim);
 *
 * This is error-prone:
 * - Easy to pass the wrong size
 * - No type safety (void* everywhere)
 * - Can't use range-based for loops
 * - Size must be tracked separately
 *
 * WITH SPAN:
 *
 *   void process_shape(Span<const int64_t> shape);
 *
 *   // Call with array - size deduced automatically!
 *   int64_t dims[4] = {1, 3, 224, 224};
 *   process_shape(dims);
 *
 *   // Or explicitly
 *   process_shape(Span<const int64_t>(ptr, 4));
 *
 *   // Inside function - use range-based for!
 *   for (int64_t dim : shape) { ... }
 *
 * =============================================================================
 * C++ CONCEPTS USED (for C programmers)
 * =============================================================================
 *
 * 1. TEMPLATES (template <typename T>)
 *    - Span<int> is a span of ints, Span<float> is a span of floats
 *    - The compiler generates specialized code for each type you use
 *    - This is resolved at compile time, no runtime cost
 *
 * 2. CONST CORRECTNESS
 *    - Span<int> allows modifying elements
 *    - Span<const int> is read-only (can't modify elements)
 *    - Always use Span<const T> when you don't need to modify
 *
 * 3. ITERATORS (begin() / end())
 *    - Pointers that mark the start and end of a sequence
 *    - Enable range-based for loops: for (auto x : span) { ... }
 *    - begin() returns pointer to first element
 *    - end() returns pointer to ONE PAST the last element
 *
 * 4. TEMPLATE PARAMETER DEDUCTION
 *    - When you pass an array, the compiler figures out N automatically
 *    - Span(T (&arr)[N]) means "reference to array of N elements"
 *    - The compiler deduces N from the array you pass
 *
 * 5. SIZE_T
 *    - An unsigned integer type for sizes and indices
 *    - Guaranteed to be large enough to hold any array size
 *    - On 32-bit systems: 4 bytes, on 64-bit systems: 8 bytes
 *
 * =============================================================================
 * MEMORY MODEL
 * =============================================================================
 *
 * NO DYNAMIC ALLOCATION. Span is just two values:
 * - A pointer to the data (doesn't own it)
 * - A size (number of elements)
 *
 * Total size: 16 bytes on 64-bit systems (8-byte pointer + 8-byte size)
 *
 * IMPORTANT: Span does NOT own the data. The data must outlive the span.
 * This is similar to how a pointer works - if you free the data, the
 * span becomes invalid (dangling).
 *
 * =============================================================================
 * USAGE EXAMPLES
 * =============================================================================
 *
 * Creating spans:
 *
 *   // From pointer and size
 *   int64_t* ptr = ...;
 *   Span<int64_t> span1(ptr, 10);
 *
 *   // From C array - size automatically deduced!
 *   int64_t arr[4] = {1, 2, 3, 4};
 *   Span<int64_t> span2(arr);  // size = 4
 *
 *   // Using helper function
 *   auto span3 = MakeSpan(ptr, 10);
 *
 * Accessing elements:
 *
 *   span[0] = 42;        // First element (no bounds check)
 *   span.front() = 1;    // First element
 *   span.back() = 99;    // Last element
 *
 * Iterating:
 *
 *   // Range-based for (preferred)
 *   for (int64_t x : span) {
 *     printf("%lld\n", x);
 *   }
 *
 *   // Index-based
 *   for (size_t i = 0; i < span.size(); i++) {
 *     printf("%lld\n", span[i]);
 *   }
 *
 * Subspans:
 *
 *   Span<int64_t> first_two = span.first(2);   // [0, 1]
 *   Span<int64_t> last_two = span.last(2);     // [2, 3]
 *   Span<int64_t> middle = span.subspan(1, 2); // [1, 2]
 *
 * =============================================================================
 */

#ifndef TVM_DSP_RUNTIME_CPP_SPAN_H_
#define TVM_DSP_RUNTIME_CPP_SPAN_H_

#include <cstddef>      /* size_t */
#include <type_traits>  /* std::enable_if, std::is_convertible, std::is_same */

namespace tvm {
namespace dsp {

/*!
 * \brief Non-owning view over a contiguous sequence of elements
 *
 * Span provides a safe, lightweight way to pass arrays around without
 * copying data. It's essentially a pointer + size bundled together.
 *
 * \tparam T The element type. Use const T for read-only access.
 *
 * LIFETIME WARNING:
 * The span does NOT own the data it points to. You must ensure the
 * underlying data outlives the span. Accessing a span after the
 * data has been freed results in undefined behavior.
 *
 * THREAD SAFETY:
 * Span itself is not thread-safe. If multiple threads access the same
 * span or underlying data, you must provide external synchronization.
 */
template <typename T>
class Span {
 public:
  /*
   * =========================================================================
   * TYPE ALIASES
   * =========================================================================
   * These are standard names that make Span work with STL algorithms
   * and range-based for loops.
   */

  using element_type = T;           /*!< The type of elements in the span */
  using value_type = T;             /*!< Same as element_type for non-const T */
  using size_type = size_t;         /*!< Type used for sizes and indices */
  using difference_type = ptrdiff_t;/*!< Type for pointer differences */
  using pointer = T*;               /*!< Pointer to element */
  using const_pointer = const T*;   /*!< Const pointer to element */
  using reference = T&;             /*!< Reference to element */
  using const_reference = const T&; /*!< Const reference to element */
  using iterator = T*;              /*!< Iterator type (just a pointer) */
  using const_iterator = const T*;  /*!< Const iterator type */

  /*
   * =========================================================================
   * CONSTRUCTORS
   * =========================================================================
   */

  /*!
   * \brief Default constructor - creates an empty span
   *
   * An empty span has data() == nullptr and size() == 0.
   * It's safe to iterate over (the loop body never executes).
   */
  Span() : data_(nullptr), size_(0) {}

  /*!
   * \brief Construct from pointer and size
   *
   * \param data Pointer to the first element
   * \param size Number of elements in the span
   *
   * WARNING: You must ensure that [data, data + size) is a valid range.
   * The span does not check this at construction time.
   *
   * Example:
   *   int64_t* arr = get_array();
   *   Span<int64_t> span(arr, 10);  // View of 10 elements starting at arr
   */
  Span(T* data, size_t size) : data_(data), size_(size) {}

  /*!
   * \brief Conversion constructor: Span<T> -> Span<const T>
   *
   * \tparam U The source element type (must be non-const version of T)
   * \param other The span to convert from
   *
   * This allows implicit conversion from Span<int> to Span<const int>.
   * The template is enabled only when U* is convertible to T* (e.g., int* -> const int*).
   *
   * Example:
   *   Span<int> mutable_span(data, 10);
   *   Span<const int> const_span = mutable_span;  // OK, implicit conversion
   *
   * NOTE: This uses SFINAE (enable_if) to only enable the constructor when
   * the conversion is valid. Don't worry if you don't understand the template
   * magic - just know it lets you pass Span<T> where Span<const T> is expected.
   */
  template <typename U,
            typename = typename std::enable_if<
                std::is_convertible<U*, T*>::value &&
                !std::is_same<U, T>::value>::type>
  Span(const Span<U>& other) : data_(other.data()), size_(other.size()) {}

  /*!
   * \brief Construct from a C-style array with automatic size deduction
   *
   * \tparam N The size of the array (automatically deduced by compiler)
   * \param arr Reference to a C-style array
   *
   * This constructor uses template parameter deduction to automatically
   * figure out the array size. You don't need to specify N manually.
   *
   * Example:
   *   int64_t dims[4] = {1, 3, 224, 224};
   *   Span<int64_t> span(dims);  // N=4 deduced automatically
   *   // span.size() == 4
   *
   * NOTE: This only works with actual arrays, not pointers!
   *   int64_t* ptr = dims;
   *   Span<int64_t> span(ptr);  // ERROR! Can't deduce size from pointer
   */
  template <size_t N>
  Span(T (&arr)[N]) : data_(arr), size_(N) {}

  /*
   * =========================================================================
   * ELEMENT ACCESS
   * =========================================================================
   */

  /*!
   * \brief Get pointer to the underlying data
   * \return Pointer to the first element, or nullptr if empty
   *
   * Use this when you need to pass the data to C functions that
   * expect a raw pointer.
   */
  T* data() const { return data_; }

  /*!
   * \brief Get the number of elements
   * \return Number of elements in the span
   */
  size_t size() const { return size_; }

  /*!
   * \brief Check if the span is empty
   * \return true if size() == 0
   */
  bool empty() const { return size_ == 0; }

  /*!
   * \brief Access element by index (no bounds checking)
   *
   * \param i Index of the element (0-based)
   * \return Reference to the element at index i
   *
   * WARNING: No bounds checking! Accessing out-of-bounds indices
   * results in undefined behavior. Use this when you're sure the
   * index is valid, for maximum performance.
   *
   * Example:
   *   span[0] = 42;  // Set first element
   *   int x = span[1];  // Get second element
   */
  T& operator[](size_t i) const { return data_[i]; }

  /*!
   * \brief Access first element
   * \return Reference to the first element
   *
   * WARNING: Calling on an empty span is undefined behavior.
   */
  T& front() const { return data_[0]; }

  /*!
   * \brief Access last element
   * \return Reference to the last element
   *
   * WARNING: Calling on an empty span is undefined behavior.
   */
  T& back() const { return data_[size_ - 1]; }

  /*
   * =========================================================================
   * ITERATORS
   * =========================================================================
   * These enable range-based for loops and compatibility with STL algorithms.
   *
   * For Span, iterators are just pointers. This is the simplest and most
   * efficient implementation for contiguous data.
   */

  /*!
   * \brief Get iterator to the first element
   * \return Pointer to the first element
   *
   * For an empty span, begin() == end().
   */
  T* begin() const { return data_; }

  /*!
   * \brief Get iterator to one past the last element
   * \return Pointer to one past the last element
   *
   * WARNING: Do not dereference end()! It points past the valid data.
   * It's only used to detect when iteration should stop.
   */
  T* end() const { return data_ + size_; }

  /*
   * =========================================================================
   * SUBSPAN OPERATIONS
   * =========================================================================
   * These create new spans that view a portion of the original data.
   * No data is copied - subspans share the underlying storage.
   */

  /*!
   * \brief Get a subspan of the first N elements
   *
   * \param count Number of elements in the subspan
   * \return Span viewing the first 'count' elements
   *
   * WARNING: count must be <= size(). No bounds checking is performed.
   *
   * Example:
   *   Span<int> full = ...;  // [0, 1, 2, 3, 4]
   *   Span<int> head = full.first(3);  // [0, 1, 2]
   */
  Span<T> first(size_t count) const {
    return Span<T>(data_, count);
  }

  /*!
   * \brief Get a subspan of the last N elements
   *
   * \param count Number of elements in the subspan
   * \return Span viewing the last 'count' elements
   *
   * WARNING: count must be <= size(). No bounds checking is performed.
   *
   * Example:
   *   Span<int> full = ...;  // [0, 1, 2, 3, 4]
   *   Span<int> tail = full.last(2);  // [3, 4]
   */
  Span<T> last(size_t count) const {
    return Span<T>(data_ + (size_ - count), count);
  }

  /*!
   * \brief Get a subspan starting at offset with given count
   *
   * \param offset Starting index of the subspan
   * \param count Number of elements (default: all remaining elements)
   * \return Span viewing elements [offset, offset + count)
   *
   * WARNING: offset + count must be <= size(). No bounds checking.
   *
   * Example:
   *   Span<int> full = ...;  // [0, 1, 2, 3, 4]
   *   Span<int> mid = full.subspan(1, 3);  // [1, 2, 3]
   *   Span<int> rest = full.subspan(2);    // [2, 3, 4]
   */
  Span<T> subspan(size_t offset, size_t count = static_cast<size_t>(-1)) const {
    if (count == static_cast<size_t>(-1)) {
      count = size_ - offset;
    }
    return Span<T>(data_ + offset, count);
  }

  /*
   * =========================================================================
   * SIZE IN BYTES
   * =========================================================================
   */

  /*!
   * \brief Get the size of the span in bytes
   * \return size() * sizeof(T)
   *
   * Useful when you need to pass the byte size to memcpy, DMA, etc.
   */
  size_t size_bytes() const { return size_ * sizeof(T); }

 private:
  T* data_;      /*!< Pointer to the first element (not owned) */
  size_t size_;  /*!< Number of elements */
};

/*
 * =============================================================================
 * HELPER FUNCTIONS
 * =============================================================================
 */

/*!
 * \brief Create a span with automatic type deduction
 *
 * \tparam T Element type (automatically deduced)
 * \param data Pointer to the first element
 * \param size Number of elements
 * \return Span<T> viewing the specified range
 *
 * This is useful when you don't want to spell out the type:
 *   auto span = MakeSpan(some_pointer, 10);
 *
 * Instead of:
 *   Span<SomeVeryLongTypeName> span(some_pointer, 10);
 */
template <typename T>
Span<T> MakeSpan(T* data, size_t size) {
  return Span<T>(data, size);
}

/*!
 * \brief Create a span from a C-style array
 *
 * \tparam T Element type (automatically deduced)
 * \tparam N Array size (automatically deduced)
 * \param arr Reference to a C-style array
 * \return Span<T> viewing the entire array
 *
 * Example:
 *   int64_t dims[4] = {1, 3, 224, 224};
 *   auto span = MakeSpan(dims);  // Span<int64_t> with size 4
 */
template <typename T, size_t N>
Span<T> MakeSpan(T (&arr)[N]) {
  return Span<T>(arr, N);
}

/*!
 * \brief Create a const span from a C-style array
 *
 * \tparam T Element type (automatically deduced)
 * \tparam N Array size (automatically deduced)
 * \param arr Reference to a C-style array
 * \return Span<const T> viewing the entire array (read-only)
 *
 * Use this when you want to prevent modification of the elements.
 */
template <typename T, size_t N>
Span<const T> MakeConstSpan(const T (&arr)[N]) {
  return Span<const T>(arr, N);
}

/*!
 * \brief Create a const span from pointer and size
 *
 * \tparam T Element type (automatically deduced)
 * \param data Pointer to the first element
 * \param size Number of elements
 * \return Span<const T> viewing the specified range (read-only)
 */
template <typename T>
Span<const T> MakeConstSpan(const T* data, size_t size) {
  return Span<const T>(data, size);
}

}  // namespace dsp
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_SPAN_H_ */
