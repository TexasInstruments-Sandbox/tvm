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
 * \file cpp/fixed_vector.h
 * \brief Fixed-capacity vector with no dynamic allocation
 *
 * =============================================================================
 * OVERVIEW
 * =============================================================================
 *
 * FixedVector is a container with a fixed maximum capacity that stores all
 * elements inline (no heap allocation). Think of it as a safer, more convenient
 * array with a dynamic size but static capacity.
 *
 * PROBLEM IT SOLVES:
 *
 * In C, you often see code like this:
 *
 *   #define MAX_DIMS 16
 *   int64_t shape[MAX_DIMS];
 *   int ndim = 0;
 *
 *   // Add elements
 *   shape[ndim++] = 32;
 *   shape[ndim++] = 64;
 *   // Easy to forget bounds check!
 *
 * Problems with this approach:
 * - Size must be tracked separately (error-prone)
 * - Easy to overflow without bounds checking
 * - Can't use range-based for loops
 * - Capacity is a magic constant scattered throughout code
 *
 * WITH FIXED_VECTOR:
 *
 *   FixedVector<int64_t, 16> shape;
 *
 *   // Add elements with automatic bounds check
 *   shape.push_back(32);  // Returns true if successful
 *   shape.push_back(64);
 *
 *   // Size is tracked automatically
 *   printf("ndim = %zu\n", shape.size());
 *
 *   // Range-based for loops work
 *   for (int64_t dim : shape) {
 *     printf("%lld\n", dim);
 *   }
 *
 * =============================================================================
 * C++ CONCEPTS USED (for C programmers)
 * =============================================================================
 *
 * 1. TEMPLATES WITH SIZE (template <typename T, size_t kCapacity>)
 *    - T is the element type (like int64_t or float)
 *    - kCapacity is the maximum number of elements (a compile-time constant)
 *    - FixedVector<int, 10> and FixedVector<int, 20> are different types!
 *
 * 2. STATIC_ASSERT
 *    - A compile-time check that fails with an error if the condition is false
 *    - Used here to ensure elements are trivially destructible
 *    - Example: static_assert(sizeof(int) == 4, "int must be 4 bytes")
 *
 * 3. TYPE TRAITS (std::is_trivially_destructible)
 *    - Compile-time type introspection
 *    - is_trivially_destructible<T>::value is true for types like int, float
 *    - It's false for types that need cleanup (like std::string)
 *    - We only support trivially destructible types to avoid destructor issues
 *
 * 4. CONSTEXPR
 *    - A function that can be evaluated at compile time
 *    - capacity() is constexpr because kCapacity is known at compile time
 *    - The compiler can optimize better when values are known at compile time
 *
 * 5. INITIALIZER LIST
 *    - Allows construction with braces: FixedVector<int, 4> v = {1, 2, 3};
 *    - std::initializer_list<T> is a lightweight view of the brace contents
 *
 * =============================================================================
 * MEMORY MODEL
 * =============================================================================
 *
 * NO DYNAMIC ALLOCATION. All storage is inline:
 *
 *   FixedVector<int64_t, 4> vec;
 *
 *   Memory layout:
 *   +------------------------------------------+
 *   | data_[0] | data_[1] | data_[2] | data_[3] | size_ |
 *   +------------------------------------------+
 *   |<------------ inline storage ------------>|
 *
 *   Total size: kCapacity * sizeof(T) + sizeof(size_t)
 *
 * The array is always fully allocated, but only size_ elements are "active".
 * This means:
 * - No allocator needed
 * - Safe for stack allocation
 * - Works on embedded systems with no heap
 * - Predictable memory footprint
 *
 * =============================================================================
 * COMPARISON: FixedVector vs std::vector vs raw array
 * =============================================================================
 *
 * | Feature           | raw array    | std::vector  | FixedVector        |
 * |-------------------|--------------|--------------|--------------------|
 * | Heap allocation   | No           | Yes          | No                 |
 * | Dynamic size      | No           | Yes          | Yes (up to cap)    |
 * | Bounds checking   | No           | Optional     | Yes (push_back)    |
 * | Range-based for   | Needs tricks | Yes          | Yes                |
 * | Auto size track   | No           | Yes          | Yes                |
 * | Embedded-safe     | Yes          | No           | Yes                |
 * | Capacity growth   | N/A          | Auto         | No                 |
 *
 * =============================================================================
 * USAGE EXAMPLES
 * =============================================================================
 *
 * Creating vectors:
 *
 *   // Empty vector with capacity 8
 *   FixedVector<int64_t, 8> vec1;
 *
 *   // From initializer list
 *   FixedVector<int, 4> vec2 = {1, 2, 3};  // size=3, capacity=4
 *
 * Adding elements:
 *
 *   FixedVector<float, 4> vec;
 *   vec.push_back(1.0f);  // Returns true
 *   vec.push_back(2.0f);  // Returns true
 *   vec.push_back(3.0f);  // Returns true
 *   vec.push_back(4.0f);  // Returns true
 *   vec.push_back(5.0f);  // Returns FALSE! Vector is full
 *
 * Accessing elements:
 *
 *   vec[0] = 42;         // Modify first element
 *   float x = vec[1];    // Read second element
 *   float& first = vec.front();
 *   float& last = vec.back();
 *
 * Iterating:
 *
 *   // Range-based for (preferred)
 *   for (float f : vec) {
 *     printf("%f\n", f);
 *   }
 *
 *   // Index-based
 *   for (size_t i = 0; i < vec.size(); i++) {
 *     printf("%f\n", vec[i]);
 *   }
 *
 * Converting to Span:
 *
 *   FixedVector<int, 8> vec = {1, 2, 3, 4};
 *   Span<int> span = vec.AsSpan();  // Non-owning view
 *
 * =============================================================================
 * WHY TRIVIALLY DESTRUCTIBLE ONLY?
 * =============================================================================
 *
 * We restrict to trivially destructible types (int, float, pointers, etc.)
 * because:
 *
 * 1. No destructor calls needed when the vector is destroyed
 * 2. No destructor calls needed when elements are removed
 * 3. Simple implementation without complex lifetime management
 * 4. Works correctly on embedded systems without runtime support
 *
 * Types that ARE trivially destructible:
 * - All primitive types (int, float, double, etc.)
 * - Pointers (int*, void*, etc.)
 * - Enums and enum classes
 * - POD structs containing only trivially destructible members
 *
 * Types that are NOT trivially destructible:
 * - std::string (needs to free memory)
 * - std::vector (needs to free memory)
 * - Any class with a destructor
 *
 * =============================================================================
 */

#ifndef TVM_DSP_RUNTIME_CPP_FIXED_VECTOR_H_
#define TVM_DSP_RUNTIME_CPP_FIXED_VECTOR_H_

#include <cstddef>          /* size_t */
#include <cstring>          /* memcpy */
#include <initializer_list> /* std::initializer_list */
#include <type_traits>      /* std::is_trivially_destructible */

/* Include Span for AsSpan() method */
#include "span.h"

namespace tvm {
namespace dsp {

/*!
 * \brief Fixed-capacity vector with inline storage
 *
 * A container that holds up to kCapacity elements without heap allocation.
 * All storage is embedded within the object itself.
 *
 * \tparam T The element type. Must be trivially destructible (no cleanup needed).
 * \tparam kCapacity The maximum number of elements. Must be > 0.
 *
 * THREAD SAFETY:
 * FixedVector is not thread-safe. If multiple threads access the same
 * vector, you must provide external synchronization.
 *
 * EXCEPTION SAFETY:
 * This class does not throw exceptions. Operations that could fail
 * (like push_back on a full vector) return bool to indicate success.
 */
template <typename T, size_t kCapacity>
class FixedVector {
  /*
   * Compile-time checks:
   * - T must be trivially destructible (no destructor needed)
   * - kCapacity must be at least 1
   */
  static_assert(std::is_trivially_destructible<T>::value,
                "FixedVector only supports trivially destructible types. "
                "Types like std::string or std::vector are not allowed.");
  static_assert(kCapacity > 0, "FixedVector capacity must be at least 1");

 public:
  /*
   * =========================================================================
   * TYPE ALIASES
   * =========================================================================
   * Standard names that make FixedVector work with STL algorithms.
   */

  using value_type = T;             /*!< The type of elements */
  using size_type = size_t;         /*!< Type for sizes and indices */
  using difference_type = ptrdiff_t;/*!< Type for pointer differences */
  using reference = T&;             /*!< Reference to element */
  using const_reference = const T&; /*!< Const reference to element */
  using pointer = T*;               /*!< Pointer to element */
  using const_pointer = const T*;   /*!< Const pointer to element */
  using iterator = T*;              /*!< Iterator type (just a pointer) */
  using const_iterator = const T*;  /*!< Const iterator type */

  /*
   * =========================================================================
   * CONSTRUCTORS
   * =========================================================================
   */

  /*!
   * \brief Default constructor - creates empty vector
   *
   * Creates a vector with size 0 and capacity kCapacity.
   * All kCapacity elements are allocated but uninitialized.
   *
   * Example:
   *   FixedVector<int, 8> vec;  // size=0, capacity=8
   */
  FixedVector() : size_(0) {}

  /*!
   * \brief Construct from initializer list
   *
   * \param init Brace-enclosed list of elements
   *
   * Creates a vector with elements copied from the initializer list.
   * If the list has more elements than kCapacity, only the first
   * kCapacity elements are copied.
   *
   * Example:
   *   FixedVector<int, 4> vec = {1, 2, 3};  // size=3, capacity=4
   *   FixedVector<int, 2> vec2 = {1, 2, 3, 4};  // size=2! Truncated!
   *
   * NOTE: If the initializer has too many elements, they are silently
   * truncated. In debug builds, you might want to add an assertion.
   */
  FixedVector(std::initializer_list<T> init) : size_(0) {
    for (const T& item : init) {
      if (size_ >= kCapacity) break;  /* Truncate if too many */
      data_[size_++] = item;
    }
  }

  /*!
   * \brief Copy constructor
   *
   * \param other The vector to copy from
   *
   * Creates an exact copy of another FixedVector of the same type and capacity.
   *
   * NOTE: This uses element-by-element copy, not memcpy, to be safe for
   * types that have non-trivial copy semantics (though we restrict to
   * trivially destructible types, they may still have copy constructors).
   */
  FixedVector(const FixedVector& other) : size_(other.size_) {
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = other.data_[i];
    }
  }

  /*!
   * \brief Copy assignment
   *
   * \param other The vector to copy from
   * \return Reference to this vector
   */
  FixedVector& operator=(const FixedVector& other) {
    if (this != &other) {
      size_ = other.size_;
      for (size_t i = 0; i < size_; ++i) {
        data_[i] = other.data_[i];
      }
    }
    return *this;
  }

  /*
   * =========================================================================
   * CAPACITY
   * =========================================================================
   */

  /*!
   * \brief Get number of elements currently in the vector
   * \return Current number of elements (0 to kCapacity)
   */
  size_t size() const { return size_; }

  /*!
   * \brief Get maximum capacity
   * \return The template parameter kCapacity
   *
   * This is constexpr because the capacity is known at compile time.
   */
  constexpr size_t capacity() const { return kCapacity; }

  /*!
   * \brief Check if vector is empty
   * \return true if size() == 0
   */
  bool empty() const { return size_ == 0; }

  /*!
   * \brief Check if vector is full
   * \return true if size() == capacity()
   *
   * When full, push_back() will fail and return false.
   */
  bool full() const { return size_ == kCapacity; }

  /*
   * =========================================================================
   * ELEMENT ACCESS
   * =========================================================================
   */

  /*!
   * \brief Access element by index (no bounds checking)
   *
   * \param i Index of the element (0-based)
   * \return Reference to the element at index i
   *
   * WARNING: No bounds checking! Accessing an index >= size() is undefined
   * behavior. Use this when you're certain the index is valid.
   *
   * Example:
   *   vec[0] = 42;
   *   int x = vec[1];
   */
  T& operator[](size_t i) { return data_[i]; }

  /*!
   * \brief Access element by index (const version)
   */
  const T& operator[](size_t i) const { return data_[i]; }

  /*!
   * \brief Get pointer to underlying array
   * \return Pointer to the first element
   *
   * Use this when you need to pass the data to C functions.
   */
  T* data() { return data_; }

  /*!
   * \brief Get const pointer to underlying array
   */
  const T* data() const { return data_; }

  /*!
   * \brief Access first element
   * \return Reference to the first element
   *
   * WARNING: Calling on empty vector is undefined behavior.
   */
  T& front() { return data_[0]; }
  const T& front() const { return data_[0]; }

  /*!
   * \brief Access last element
   * \return Reference to the last element
   *
   * WARNING: Calling on empty vector is undefined behavior.
   */
  T& back() { return data_[size_ - 1]; }
  const T& back() const { return data_[size_ - 1]; }

  /*
   * =========================================================================
   * MODIFIERS
   * =========================================================================
   */

  /*!
   * \brief Add element to the end
   *
   * \param item The element to add
   * \return true if successful, false if vector is full
   *
   * This is the safe way to add elements - always check the return value!
   *
   * Example:
   *   if (!vec.push_back(42)) {
   *     // Handle full vector
   *   }
   */
  bool push_back(const T& item) {
    if (size_ >= kCapacity) return false;
    data_[size_++] = item;
    return true;
  }

  /*!
   * \brief Remove last element
   *
   * Decrements size by 1. The element is not destroyed (just inaccessible).
   *
   * WARNING: Calling on empty vector is undefined behavior.
   *
   * Example:
   *   vec.push_back(1);
   *   vec.push_back(2);
   *   vec.pop_back();  // Now vec = {1}
   */
  void pop_back() {
    if (size_ > 0) --size_;
  }

  /*!
   * \brief Remove all elements
   *
   * Sets size to 0. Elements are not destroyed (just inaccessible).
   * Capacity is unchanged.
   */
  void clear() { size_ = 0; }

  /*!
   * \brief Resize the vector
   *
   * \param new_size The new size
   * \return true if successful, false if new_size > capacity()
   *
   * If new_size < size(), elements are removed from the end.
   * If new_size > size(), new elements are default-initialized (zeroed
   * for arithmetic types).
   *
   * Example:
   *   vec.resize(5);  // Exactly 5 elements now
   */
  bool resize(size_t new_size) {
    if (new_size > kCapacity) return false;
    /* If growing, zero-initialize new elements */
    for (size_t i = size_; i < new_size; ++i) {
      data_[i] = T{};
    }
    size_ = new_size;
    return true;
  }

  /*!
   * \brief Resize with fill value
   *
   * \param new_size The new size
   * \param value The value to fill new elements with
   * \return true if successful, false if new_size > capacity()
   */
  bool resize(size_t new_size, const T& value) {
    if (new_size > kCapacity) return false;
    /* Fill new elements with value */
    for (size_t i = size_; i < new_size; ++i) {
      data_[i] = value;
    }
    size_ = new_size;
    return true;
  }

  /*
   * =========================================================================
   * ITERATORS
   * =========================================================================
   * These enable range-based for loops and STL algorithm compatibility.
   */

  /*!
   * \brief Get iterator to first element
   * \return Pointer to the first element
   */
  T* begin() { return data_; }
  const T* begin() const { return data_; }

  /*!
   * \brief Get iterator to one past last element
   * \return Pointer to one past the last element
   *
   * WARNING: Do not dereference end()!
   */
  T* end() { return data_ + size_; }
  const T* end() const { return data_ + size_; }

  /*
   * =========================================================================
   * CONVERSION
   * =========================================================================
   */

  /*!
   * \brief Get a Span view of the vector contents
   * \return Span<T> viewing all elements in the vector
   *
   * The span is only valid as long as the vector exists and is not
   * modified (adding/removing elements invalidates the span).
   *
   * Example:
   *   FixedVector<int, 8> vec = {1, 2, 3};
   *   Span<int> span = vec.AsSpan();
   *   // span now views {1, 2, 3}
   */
  Span<T> AsSpan() { return Span<T>(data_, size_); }

  /*!
   * \brief Get a const Span view of the vector contents
   * \return Span<const T> viewing all elements (read-only)
   */
  Span<const T> AsSpan() const { return Span<const T>(data_, size_); }

 private:
  T data_[kCapacity];  /*!< Inline storage for elements */
  size_t size_;        /*!< Current number of elements */
};

/*
 * =============================================================================
 * HELPER FUNCTIONS
 * =============================================================================
 */

/*!
 * \brief Create a FixedVector from an initializer list with explicit capacity
 *
 * \tparam kCapacity The capacity of the resulting vector
 * \tparam T Element type (automatically deduced)
 * \param init The initializer list
 * \return FixedVector<T, kCapacity> containing the elements
 *
 * Example:
 *   auto vec = MakeFixedVector<8>({1, 2, 3, 4});
 *   // vec is FixedVector<int, 8> with size 4
 */
template <size_t kCapacity, typename T>
FixedVector<T, kCapacity> MakeFixedVector(std::initializer_list<T> init) {
  return FixedVector<T, kCapacity>(init);
}

}  // namespace dsp
}  // namespace tvm

#endif  /* TVM_DSP_RUNTIME_CPP_FIXED_VECTOR_H_ */
