// A simple and lightweight dynamic matrix implementation.
#ifndef MATRIX_H_
#define MATRIX_H_

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>

template <typename T>
class Matrix {
 public:
  // Construction.
  Matrix(uint32_t rows, uint32_t cols, bool is_row_major = true);
  Matrix(uint32_t rows, uint32_t cols, const T& value,
         bool is_row_major = true);
  Matrix(uint32_t rows, uint32_t cols, std::initializer_list<T> l,
         bool is_row_major = true);

  Matrix(const Matrix& other, bool is_row_major = true);
  template <typename X>
  Matrix(const Matrix<X>& other, bool is_row_major = true);
  template <typename X>
  Matrix& operator=(const Matrix<X>& other);

  Matrix(Matrix&& other);
  Matrix& operator=(Matrix&& other);

  // Member access.
  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  uint32_t rows() const { return rows_; }
  uint32_t cols() const { return cols_; }
  uint32_t size() const { return rows_ * cols_; }
  bool is_row_major() const { return is_row_major_; }

  // Element Access.
  
  // Non-bounds-checked. Behavior undefined if index is out of range.
  T& operator()(uint32_t i, uint32_t j);
  const T& operator()(uint32_t i, uint32_t j) const;

  // These are direct indexes into the array, so the result will be same
  // regardless of row/col majorness.
  T& operator()(uint32_t l);
  const T& operator()(uint32_t l) const;

  // Bounds-checked. Raises std::out_of_range if index is out of range.
  T& at(uint32_t i, uint32_t j);
  const T& at(uint32_t i, uint32_t j) const;
  T& at(uint32_t l);
  const T& at(uint32_t l) const;

  // Array-like arithmetic operations. These can broadcast just like NumPy.
  // Raises std::invalid_argument if shapes do not match.
  template <typename X>
  Matrix& operator+=(const X& rhs);
  template <typename X>
  Matrix& operator-=(const X& rhs);
  template <typename X>
  Matrix& operator*=(const X& rhs);
  template <typename X>
  Matrix& operator/=(const X& rhs);

  //// Matrix operations.

  // Computes matrix product with rhs.
  template <typename X>
  Matrix dot(const Matrix<X>& rhs) const;

  // Transpose the matrix. This will not copy any memory, i.e. the transposed
  // matrix will refer to the same memory as the original. A copy can be made
  // using the copy constructor.
  Matrix transpose();

 private:
  // This can be shared between transposed versions of the same matrix.
  std::shared_ptr<T> data_;
  uint32_t rows_, cols_;
  // Stores whether the data is in row or column-major order. When transposing
  // the matrix, this should be inverted.
  bool is_row_major_;

  Matrix(std::shared_ptr<T> data, uint32_t rows, uint32_t cols,
         bool is_row_major);
};

#include "matrix_impl.h"

#endif  // MATRIX_H_
