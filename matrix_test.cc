#include <functional>
#include <iostream>

#include "catch.hpp"
#include "matrix.h"

template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::plus<T>) {
  return lhs + rhs;
}
                         
template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::minus<T>) {
  return lhs - rhs;
}
                         
template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::multiplies<T>) {
  return lhs * rhs;
}
                         
template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::divides<T>) {
  return lhs / rhs;
}
                         
template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::equal_to<T>) {
  return lhs == rhs;
}
                         
template <typename T, typename U>
Matrix<T> apply_binary_op(const Matrix<T>& lhs, const Matrix<U>& rhs,
                          std::not_equal_to<T>) {
  return lhs != rhs;
}
                         
struct TestOpts {
  bool no_zeros = false;
};

template <typename F>
void test_binary_op(TestOpts opts = TestOpts{}) {
  F func;

  constexpr size_t kRows = 10;
  constexpr size_t kCols = 10;

  Matrix<typename F::first_argument_type> m1(kRows, kCols);
  Matrix<typename F::second_argument_type> m2(kRows, kCols);

  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      m1(i, j) = i * i + j;
      m2(i, j) = j * j + i;

      if (opts.no_zeros) {
        ++m1(i, j);
        ++m2(i, j);
      }
    }
  }
  const auto result = apply_binary_op(m1, m2, F());
  const auto result_transpose =
      apply_binary_op(m1.transpose(), m2.transpose(), F());

  Matrix<typename F::result_type> expected(kRows, kCols);
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      REQUIRE(result(i, j) == func(m1(i, j), m2(i, j)));
      REQUIRE(result_transpose(j, i) ==
              func(m1.transpose()(j, i), m2.transpose()(j, i)));
    }
  }
}

TEST_CASE("Plus", "[binary_op]") {
  test_binary_op<std::plus<int>>();
  test_binary_op<std::plus<float>>();
}

TEST_CASE("Minus", "[binary_op]") {
  test_binary_op<std::minus<int>>();
  test_binary_op<std::minus<float>>();
}

TEST_CASE("Multiplies", "[binary_op]") {
  test_binary_op<std::multiplies<int>>();
  test_binary_op<std::multiplies<float>>();
}

TEST_CASE("Divides", "[binary_op]") {
  TestOpts opts;
  opts.no_zeros = true;
  test_binary_op<std::divides<int>>(opts);
  test_binary_op<std::divides<float>>(opts);
}

TEST_CASE("EqualTo", "[binary_op]") {
  test_binary_op<std::equal_to<int>>();
  test_binary_op<std::equal_to<float>>();
}

TEST_CASE("NotEqualTo", "[binary_op]") {
  test_binary_op<std::not_equal_to<int>>();
  test_binary_op<std::not_equal_to<float>>();
}

TEST_CASE("PlusZeroSizeMatrix", "[zero_matrix]") {
  Matrix<int> m1(0, 5);
  Matrix<int> m2(0, 5);
  const auto result = m1 + m2;
  const auto result_transpose = m1.transpose() + m2.transpose();
  REQUIRE(result.rows() == 0);
  REQUIRE(result.cols() == 5);
  REQUIRE(result_transpose.rows() == 5);
  REQUIRE(result_transpose.cols() == 0);
}

TEST_CASE("All", "[all]") {
  Matrix<int> m1(2, 2, {1, 2, 3, 4});
  REQUIRE(m1.all());
  m1(0, 0) = 0;
  REQUIRE(!m1.all());
}

TEST_CASE("Any", "[any]") {
  Matrix<int> m1(2, 2, {0, 0, 0, 4});
  REQUIRE(m1.any());
  m1(0, 3) = 0;
  REQUIRE(!m1.any());
}

TEST_CASE("DotZeroSizeMatrix", "[zero_matrix]") {
  Matrix<int> m1(0, 5);
  Matrix<int> m2(5, 0);
  const auto result = m1.dot(m2);
  const auto result_transpose = m1.transpose().dot(m2.transpose());
  REQUIRE(result.rows() == 0);
  REQUIRE(result.cols() == 0);
  REQUIRE(result_transpose.rows() == 5);
  REQUIRE(result_transpose.cols() == 5);
  REQUIRE((result_transpose == 0).all());
}

TEST_CASE("DotRowRowMajor", "[dot]") {
  Matrix<int> m1(2, 2, {1, 2,
                        3, 4});
  Matrix<int> m2(2, 3, {1, 2, 3,
                        4, 5, 6});
  Matrix<int> expected(2, 3, {9, 12, 15,
                              19, 26, 33});
  REQUIRE((m1.dot(m2) == expected).all());
}

TEST_CASE("DotRowColMajor", "[dot]") {
  constexpr double kNotRowMajor = false;
  Matrix<int> m1(2, 2, {1, 2,
                        3, 4});
  Matrix<int> m2(2, 3, {1, 4,
                        2, 5,
                        3, 6},
                 kNotRowMajor);
  Matrix<int> expected(2, 3, {9, 12, 15,
                              19, 26, 33});
  REQUIRE((m1.dot(m2) == expected).all());
}

TEST_CASE("DotColRowMajor", "[dot]") {
  constexpr double kNotRowMajor = false;
  Matrix<int> m1(2, 2, {1, 3,
                        2, 4},
                 kNotRowMajor);
  Matrix<int> m2(2, 3, {1, 2, 3,
                        4, 5, 6});
  Matrix<int> expected(2, 3, {9, 12, 15,
                              19, 26, 33});
  REQUIRE((m1.dot(m2) == expected).all());
}

TEST_CASE("DotColColMajor", "[dot]") {
  constexpr double kNotRowMajor = false;
  Matrix<int> m1(2, 2, {1, 3,
                        2, 4},
                 kNotRowMajor);
  Matrix<int> m2(2, 3, {1, 4,
                        2, 5,
                        3, 6},
                 kNotRowMajor);
  Matrix<int> expected(2, 3, {9, 12, 15,
                              19, 26, 33});
  REQUIRE((m1.dot(m2) == expected).all());
}
