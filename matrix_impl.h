// Implementation of matrix methods.
#include <algorithm>
#include <cassert>
#include <functional>
#include <type_traits>

namespace internal {

// Performs the bounds check. Raises std::out_of_range exception if row/col is
// out of range.
template <typename T>
void check_bounds(const Matrix<T>& m, uint32_t i, uint32_t j) {
  if (i >= m.rows()) {
    throw std::out_of_range("Row index is out of range");
  } else if (j >= m.cols()) {
    throw std::out_of_range("Column index is out of range");
  }
}

template <typename T>
void check_bounds(const Matrix<T>& m, uint32_t l) {
  if (l >= m.rows() * m.cols()) {
    throw std::out_of_range("Linear index is out of range");
  }
}

// Check that matrix dimensions match. Raises std::invalid_argument exception
// if they don't match.
template <typename T, typename U>
void check_dim_match(const T& lhs, const U& rhs) {
  if (lhs.rows() != rhs.rows()) {
    throw std::invalid_argument("Row dimension doesn't match");
  } else if (lhs.cols() != rhs.cols()) {
    throw std::invalid_argument("Col dimension doesn't match");
  }
}

// Check that matrix dimensions match for matrix multiplication. Raises
// std::invalid_argument exception if they don't match.
template <typename T, typename U>
void check_dot_dim_match(const T& lhs, const U& rhs) {
  if (lhs.cols() != rhs.rows()) {
    throw std::invalid_argument("lhs column dim doesn't match rhs row dim");
  }
}

// An iterator maintains a linear index over a 2D table, and abstracts over
// different iteration orders. A "friendly" iterator means the memory is layed
// out in the same way as is being iterated over, which means we can advance to
// the next major dim by just incrementing by one. An "unfriendly" iterator
// means the memory layout is flipped from the iteration order, so advancing to
// the next major dimensions means we must rewind to the beginning of the minor
// dim and then jump to the next major dim.

class FriendlyIterator {
 public:
  FriendlyIterator(): index_(0) {}
  size_t index() const { return index_; }
  void next_minor_dim() { ++index_; }
  void rewind_minor_dim(size_t n) { index_ -= n; }
  // Only call this when we're at the end of the current minor dim.
  void next_major_dim() { ++index_; }
  void reset() { index_ = 0; }

 private:
  size_t index_;
};

class UnfriendlyIterator {
 public:
  UnfriendlyIterator(size_t minor_dim_stride):
      index_(0), minor_dim_stride_(minor_dim_stride) {}
  size_t index() const { return index_; }
  void next_minor_dim() { index_ += minor_dim_stride_; }
  void rewind_minor_dim(size_t n) { index_ -= minor_dim_stride_ * n; }
  // Only call this when we're at the end of the current minor dim.
  void next_major_dim() { index_ = (index_ + 1) % minor_dim_stride_; }
  void reset() { index_ = 0; }

 private:
  size_t index_;
  size_t minor_dim_stride_;
};

// A RowMajor iterator has minor dim equal to number of columns, and major dim
// equal to number of rows.
class RowMajorIterator {
 public:
  RowMajorIterator(size_t rows, size_t cols)
      : rows_(rows), cols_(cols) {}
  size_t minor_dim_size() const { return cols_; }
  size_t major_dim_size() const { return rows_; }

 private:
  size_t rows_, cols_;
};

// A ColMajor iterator has minor dim equal to number of rows, and major dim
// equal to number of cols.
class ColMajorIterator {
 public:
  ColMajorIterator(size_t rows, size_t cols)
      : rows_(rows), cols_(cols) {}
  size_t minor_dim_size() const { return rows_; }
  size_t major_dim_size() const { return cols_; }

 private:
  size_t rows_, cols_;
};

// Now we can define RowMajor and ColMajor iterators of the friendly and
// unfriendly nature. These all have the same interface.

class RowMajorFriendlyIterator : public FriendlyIterator,
                                 public RowMajorIterator {
 public:
  static constexpr bool is_row_major = true;
  RowMajorFriendlyIterator(size_t rows, size_t cols)
      : FriendlyIterator(), RowMajorIterator(rows, cols) {}
};

class RowMajorUnfriendlyIterator : public UnfriendlyIterator,
                                   public RowMajorIterator {
 public:
  static constexpr bool is_row_major = true;
  RowMajorUnfriendlyIterator(size_t rows, size_t cols)
      : UnfriendlyIterator(rows), RowMajorIterator(rows, cols) {}
};

class ColMajorFriendlyIterator : public FriendlyIterator,
                                 public ColMajorIterator {
 public:
  static constexpr bool is_row_major = false;
  ColMajorFriendlyIterator(size_t rows, size_t cols)
      : FriendlyIterator(), ColMajorIterator(rows, cols) {}
};

class ColMajorUnfriendlyIterator : public UnfriendlyIterator,
                                   public ColMajorIterator {
 public:
  static constexpr bool is_row_major = false;
  ColMajorUnfriendlyIterator(size_t rows, size_t cols)
      : UnfriendlyIterator(cols), ColMajorIterator(rows, cols) {}
};

// We can also define broadcast iterators, which operate on 1-dimensional
// arrays of data. A "friendly" broadcast iterator is one where the dimension
// of the matrix agrees with the iteration order. For instance, a 1 x n matrix
// broadcast as a row major iterator would be classified as friendly, but if it
// was broadcast as a col major iterator, it would be classified as unfriendly.
//
// These broadcast iterators don't support major/minor dim size queries, since
// they should always be on the rhs of an expression.

class BroadcastFriendlyIterator {
 public:
  BroadcastFriendlyIterator() : index_(0) {}
  size_t index() const { return index_; }
  void next_minor_dim() { ++index_; }
  void rewind_minor_dim(size_t n) { index_ -= n; }
  void next_major_dim() { index_ = 0; }
  void reset() { index_ = 0; }

 private:
  size_t index_;
};

class BroadcastUnfriendlyIterator {
 public:
  BroadcastUnfriendlyIterator() : index_(0) {}
  size_t index() const { return index_; }
  void next_minor_dim() {}
  void rewind_minor_dim(size_t n) {}
  void next_major_dim() { ++index_; }
  void reset() { index_ = 0; }

 private:
  size_t index_;
};

class BroadcastRowMajorFriendlyIterator : public BroadcastFriendlyIterator {
 public:
  static constexpr bool is_row_major = true;
  BroadcastRowMajorFriendlyIterator(size_t rows, size_t cols)
    : BroadcastFriendlyIterator() {}
};

class BroadcastRowMajorUnfriendlyIterator : public BroadcastUnfriendlyIterator {
 public:
  static constexpr bool is_row_major = true;
  BroadcastRowMajorUnfriendlyIterator(size_t rows, size_t cols)
    : BroadcastUnfriendlyIterator() {}
};

class BroadcastColMajorFriendlyIterator : public BroadcastFriendlyIterator {
 public:
  static constexpr bool is_row_major = false;
  BroadcastColMajorFriendlyIterator(size_t rows, size_t cols)
    : BroadcastFriendlyIterator() {}
};

class BroadcastColMajorUnfriendlyIterator : public BroadcastUnfriendlyIterator {
 public:
  static constexpr bool is_row_major = false;
  BroadcastColMajorUnfriendlyIterator(size_t rows, size_t cols)
    : BroadcastUnfriendlyIterator() {}
};

// Finally, for uniformity, we can define a shell matrix over a single value,
// and a row/col major iterator over this single-value shell.
template <typename T>
struct ShellMatrix {
  const T& val;
  const T& operator()(uint32_t) const { return val; }
  uint32_t rows() const { return 1; }
  uint32_t cols() const { return 1; }
};

template <bool IS_ROW_MAJOR>
class ShellMatrixIterator {
 public:
  static constexpr bool is_row_major = IS_ROW_MAJOR;
  ShellMatrixIterator(size_t rows, size_t cols) {}
  size_t index() const { return 0; }
  void next_minor_dim() {}
  void rewind_minor_dim(size_t) {}
  void next_major_dim() {}
  void reset() {}
};

using ShellMatrixRowMajorIterator = ShellMatrixIterator<true>;
using ShellMatrixColMajorIterator = ShellMatrixIterator<false>;

// Applies an operator between two matrix-like objects, and assigns the result
// to the matrix-like object dst. This is suffixed with "_helper" because it's
// not invoked directly.
template <typename F,
          typename TIter, typename UIter, typename VIter,
          typename T, typename U, typename V>
static void apply_operator_helper(T* dst, const U& lhs, const V& rhs) {
  static_assert(TIter::is_row_major == UIter::is_row_major &&
                TIter::is_row_major == VIter::is_row_major,
                "dst, lhs, and rhs must have same row/col majorness");
  check_dim_match(*dst, lhs);

  F functor;
  TIter dst_iter(dst->rows(), dst->cols());
  UIter lhs_iter(lhs.rows(), lhs.cols());
  VIter rhs_iter(rhs.rows(), rhs.cols());

  for (size_t maj = 0; maj < dst_iter.major_dim_size(); ++maj) {
    for (size_t min = 0; min + 1 < dst_iter.minor_dim_size(); ++min) {
      (*dst)(dst_iter.index()) = functor(
          static_cast<typename F::first_argument_type>(lhs(lhs_iter.index())),
          static_cast<typename F::second_argument_type>(rhs(rhs_iter.index())));
      dst_iter.next_minor_dim();
      lhs_iter.next_minor_dim();
      rhs_iter.next_minor_dim();
    }
    (*dst)(dst_iter.index()) = functor(
        static_cast<typename F::first_argument_type>(lhs(lhs_iter.index())),
        static_cast<typename F::second_argument_type>(rhs(rhs_iter.index())));
    dst_iter.next_major_dim();
    lhs_iter.next_major_dim();
    rhs_iter.next_major_dim();
  }
}

// Apply operator helper functions, to help ease the burden of all the
// different type configurations we need to handle.

// Apply the operator when all but the rhs is figured out. If rhs is a matrix,
// uses broadcasting rules. Otherwise it's a shell matrix.

template <bool IS_ROW_MAJOR, typename F, typename TIter, typename UIter,
          typename T, typename U, typename V>
void apply_operator_helper(T* dst, const U& lhs, const ShellMatrix<V>& rhs) {
  using VIter = typename std::conditional<IS_ROW_MAJOR,
      ShellMatrixRowMajorIterator, ShellMatrixColMajorIterator>::type;
  apply_operator_helper<F, TIter, UIter, VIter>(dst, lhs, rhs);
}

template <bool IS_ROW_MAJOR, typename F, typename TIter, typename UIter,
          typename T, typename U, typename V>
void apply_operator_helper(T* dst, const U& lhs, const Matrix<V>& rhs) {
  bool broadcast_rows;
  bool broadcast_cols;
  if (lhs.rows() == rhs.rows()) {
    broadcast_rows = false;
  } else if (rhs.rows() == 1) {
    broadcast_rows = true;
  } else {
    throw std::invalid_argument("Mismatching dimensions");
  }
  if (lhs.cols() == rhs.cols()) {
    broadcast_cols = false;
  } else if (rhs.cols() == 1) {
    broadcast_cols = true;
  } else {
    throw std::invalid_argument("Mismatching dimensions");
  }

  if (broadcast_rows && broadcast_cols) {
    // Use the ShellMatrix version of this function.
    apply_operator_helper<IS_ROW_MAJOR, F, TIter, UIter>(
        dst, lhs, ShellMatrix<V>{rhs(0)});
  } else if (broadcast_rows) {
    // rhs is a 1 x n matrix.
    using VIter = typename std::conditional<IS_ROW_MAJOR,
        BroadcastRowMajorFriendlyIterator,
        BroadcastColMajorUnfriendlyIterator>::type;
    apply_operator_helper<F, TIter, UIter, VIter>(dst, lhs, rhs);
  } else if (broadcast_cols) {
    // rhs is an n x 1 matrix.
    using VIter = typename std::conditional<IS_ROW_MAJOR,
        BroadcastRowMajorUnfriendlyIterator,
        BroadcastColMajorFriendlyIterator>::type;
    apply_operator_helper<F, TIter, UIter, VIter>(dst, lhs, rhs);
  } else if (rhs.is_row_major()) {
    using VIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorFriendlyIterator,
        ColMajorUnfriendlyIterator>::type;
    apply_operator_helper<F, TIter, UIter, VIter>(dst, lhs, rhs);
  } else {
    using VIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorUnfriendlyIterator,
        ColMajorFriendlyIterator>::type;
    apply_operator_helper<F, TIter, UIter, VIter>(dst, lhs, rhs);
  }
}

// Apply the operator when only dst has been figured out.
template <bool IS_ROW_MAJOR, typename F, typename TIter,
          typename T, typename U, typename V>
void apply_operator_helper(T* dst, const Matrix<U>& lhs, const V& rhs) {
  if (lhs.is_row_major()) {
    using UIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorFriendlyIterator,
        ColMajorUnfriendlyIterator>::type;
    apply_operator_helper<IS_ROW_MAJOR, F, TIter, UIter>(dst, lhs, rhs);
  } else {
    using UIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorUnfriendlyIterator,
        ColMajorFriendlyIterator>::type;
    apply_operator_helper<IS_ROW_MAJOR, F, TIter, UIter>(dst, lhs, rhs);
  }
}

// Apply the operator when only the row/col majorness has been figured out.
template <bool IS_ROW_MAJOR, typename F,
          typename T, typename U, typename V>
void apply_operator_helper(Matrix<T>* dst, const U& lhs, const V& rhs) {
  if (dst->is_row_major()) {
    using TIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorFriendlyIterator,
        ColMajorUnfriendlyIterator>::type;
    apply_operator_helper<IS_ROW_MAJOR, F, TIter>(dst, lhs, rhs);
  } else {
    using TIter = typename std::conditional<IS_ROW_MAJOR,
        RowMajorUnfriendlyIterator,
        ColMajorFriendlyIterator>::type;
    apply_operator_helper<IS_ROW_MAJOR, F, TIter>(dst, lhs, rhs);
  }
}

// Applies an operator between two matrices.
template <typename F, typename T, typename U, typename V>
void apply_operator(
    Matrix<T>* dst, const Matrix<U>& lhs, const Matrix<V>& rhs) {
  // Vote on the row/col-majorness based on the majority.
  const int total_row_major =
      static_cast<int>(dst->is_row_major()) +
      static_cast<int>(lhs.is_row_major()) +
      static_cast<int>(rhs.is_row_major());
  if (total_row_major >= 2) {
    apply_operator_helper<true, F>(dst, lhs, rhs);
  } else {
    apply_operator_helper<false, F>(dst, lhs, rhs);
  }
}

// Applies an operator between a matrix and a scalar. Assigns the results to
// dst.
template <typename F, typename T, typename U, typename V>
void apply_operator(Matrix<T>* dst, const Matrix<U>& lhs, const V& rhs) {
  // Always go with the row/col majorness of dst, since we're writing to it.
  if (dst->is_row_major()) {
    apply_operator_helper<true, F>(dst, lhs, ShellMatrix<V>{rhs});
  } else {
    apply_operator_helper<false, F>(dst, lhs, ShellMatrix<V>{rhs});
  }
}

// Copies data from one matrix-like object to the destination. Their dimensions
// must match exactly. dst is assumed to be uninitialized.
template <typename T, typename U>
void copy_data(Matrix<T>* dst, const Matrix<U>& src) {
  check_dim_match(*dst, src);
  if (dst->is_row_major() == src.is_row_major()) {
    // Order of storage matches, so we can do a direct copy.
    std::copy(
        src.data(), src.data() + src.size(), dst->data());
  } else {
    // Flipped order of storage, so we have to do a manual copy.
    if (dst->is_row_major()) {
      RowMajorFriendlyIterator dst_iter(dst->rows(), dst->cols());
      RowMajorUnfriendlyIterator src_iter(dst->rows(), dst->cols());
      for (size_t maj = 0; maj < dst_iter.major_dim_size(); ++maj) {
        for (size_t min = 0; min + 1 < dst_iter.minor_dim_size(); ++min) {
          (*dst)(dst_iter.index()) = src(src_iter.index());
          dst_iter.next_minor_dim();
          src_iter.next_minor_dim();
        }
        (*dst)(dst_iter.index()) = src(src_iter.index());
        dst_iter.next_major_dim();
        src_iter.next_major_dim();
      }
    } else {
      ColMajorFriendlyIterator dst_iter(dst->rows(), dst->cols());
      ColMajorUnfriendlyIterator src_iter(dst->rows(), dst->cols());
      for (size_t maj = 0; maj < dst_iter.major_dim_size(); ++maj) {
        for (size_t min = 0; min + 1 < dst_iter.minor_dim_size(); ++min) {
          (*dst)(dst_iter.index()) = src(src_iter.index());
          dst_iter.next_minor_dim();
          src_iter.next_minor_dim();
        }
        (*dst)(dst_iter.index()) = src(src_iter.index());
        dst_iter.next_major_dim();
        src_iter.next_major_dim();
      }
    }
  }
}

// Matrix multiplies lhs and rhs, and stores the result in dst. Does row-wise
// matrix-multiplication (iterate over dst in row-major order). Assumes dst and
// lhs are both row-major. Uses the given ColMajor iterator type for rhs.
template <typename UIter, typename T, typename U>
void dot_rowwise(Matrix<T>* dst, const Matrix<T>& lhs, const Matrix<U>& rhs) {
  assert(dst->is_row_major());
  assert(lhs.is_row_major());
  static_assert(!UIter::is_row_major,
                "In dot_rowwise, rhs iterator must be column-major");

  // In the special case of lhs.cols() == rhs.rows() == 0, just fill dst with
  // all 0s.
  if (lhs.cols() == 0) {
    dst->fill(T());
    return;
  }

  RowMajorFriendlyIterator lhs_iter(lhs.rows(), lhs.cols());
  UIter rhs_iter(rhs.rows(), rhs.cols());

  for (size_t dst_row = 0; dst_row < dst->rows(); ++dst_row) {
    for (size_t dst_col = 0; dst_col < dst->cols(); ++dst_col) {
      T* dst_ptr = &(*dst)(dst_row, dst_col);
      *dst_ptr = T();
      // Iterate over the current row of lhs, and the current column of rhs.
      for (size_t inner_ind = 0; inner_ind + 1 < lhs.cols(); ++inner_ind) {
        (*dst_ptr) += lhs(lhs_iter.index()) * rhs(rhs_iter.index());
        lhs_iter.next_minor_dim();
        rhs_iter.next_minor_dim();
      }
      (*dst_ptr) += lhs(lhs_iter.index()) * rhs(rhs_iter.index());

      // For the final one, if we're at the last column, move lhs to the next
      // row and reset rhs to the very beginning. Otherwise, rewind the lhs to
      // the start of the current row, and advance rhs to the next column.
      if (dst_col + 1 < dst->cols()) {
        lhs_iter.rewind_minor_dim(lhs.cols() - 1);
        rhs_iter.next_major_dim();
      } else {
        lhs_iter.next_major_dim();
        rhs_iter.reset();
      }
    }
  }
}

// Matrix multiplies lhs and rhs, and stores the result in dst. Does col-wise
// matrix-multiplication (iterate over dst in col-major order). Assumes dst and
// lhs are both col-major. Uses the given RowMajor iterator type for rhs.
template <typename UIter, typename T, typename U>
void dot_colwise(Matrix<T>* dst, const Matrix<T>& lhs, const Matrix<U>& rhs) {
  assert(!dst->is_row_major());
  assert(!lhs.is_row_major());
  static_assert(UIter::is_row_major,
                "In dot_colwise, rhs iterator must be row-major");

  // Initialize the destination to all 0s.
  dst->fill(T());

  // In the special case of lhs.rows() == 0, end it here.
  if (lhs.rows() == 0) return;

  ColMajorFriendlyIterator dst_iter(dst->rows(), dst->cols());
  ColMajorFriendlyIterator lhs_iter(lhs.rows(), lhs.cols());
  UIter rhs_iter(rhs.rows(), rhs.cols());

  for (size_t rhs_row = 0; rhs_row < rhs.rows(); ++rhs_row) {
    for (size_t rhs_col = 0; rhs_col < rhs.cols(); ++rhs_col) {
      const U& rhs_val = rhs(rhs_iter.index());
      // Iterate over the current columns of dst and lhs.
      for (size_t inner_ind = 0; inner_ind + 1 < lhs.rows(); ++inner_ind) {
        (*dst)(dst_iter.index()) += lhs(lhs_iter.index()) * rhs_val;
        dst_iter.next_minor_dim();
        lhs_iter.next_minor_dim();
      }
      (*dst)(dst_iter.index()) += lhs(lhs_iter.index()) * rhs_val;

      // For the final one, if we're at the last column of rhs, move lhs to the
      // next column, reset dst to the very beginning, and move rhs to the next
      // major dim. Otherwise, rewind lhs to the start of the current column,
      // advance dst to the next column, and move rhs to the next minor dim.
      if (rhs_col + 1 < rhs.cols()) {
        lhs_iter.rewind_minor_dim(lhs.rows() - 1);
        dst_iter.next_major_dim();
        rhs_iter.next_minor_dim();
      } else {
        lhs_iter.next_major_dim();
        dst_iter.reset();
        rhs_iter.next_major_dim();
      }
    }
  }
}


}  // namespace internal

template <typename T>
Matrix<T>::Matrix(uint32_t rows, uint32_t cols, bool is_row_major)
    : data_(new T[rows * cols]),
      rows_(rows),
      cols_(cols),
      is_row_major_(is_row_major) {}

template <typename T>
Matrix<T>::Matrix(uint32_t rows, uint32_t cols, const T& value,
                  bool is_row_major)
    : Matrix(rows, cols, is_row_major) {
  std::fill(data_.get(), data_.get() + size(), value);
}

template <typename T>
Matrix<T>::Matrix(uint32_t rows, uint32_t cols, std::initializer_list<T> l,
                  bool is_row_major)
    : Matrix(rows, cols, is_row_major) {
  std::copy(l.begin(), l.end(), data_.get());
}

template <typename T>
Matrix<T>::Matrix(const Matrix& other, bool is_row_major)
  : Matrix(other.rows(), other.cols(), is_row_major) {
  internal::copy_data(this, other);
}

template <typename T>
template <typename X>
Matrix<T>::Matrix(const Matrix<X>& other, bool is_row_major)
  : Matrix(other.rows(), other.cols(), is_row_major) {
  internal::copy_data(this, other);
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other)
  : data_(std::move(other.data_)), rows_(other.rows_),
    cols_(other.cols_), is_row_major_(other.is_row_major_) {}

template <typename T>
template <typename X>
Matrix<T>& Matrix<T>::operator=(const Matrix<X>& other) {
  if (this == &other) return *this;
  if (this->rows() != other.rows() || this->cols() != other.cols()) {
    data_.reset(new T[other.rows_ * other.cols_]);
    rows_ = other.rows();
    cols_ = other.cols();
  }
  internal::copy_data(this, other);
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) {
  if (this == &other) return *this;
  data_ = std::move(other.data_);
  rows_ = other.rows_;
  cols_ = other.cols_;
  is_row_major_ = other.is_row_major_;
  return *this;
}

template <typename T>
template <typename X>
void Matrix<T>::fill(const X& val) {
  std::fill(data(), data() + (rows() * cols()), val);
}

template <typename T>
T& Matrix<T>::operator()(uint32_t i, uint32_t j) {
  if (is_row_major()) {
    return data()[i * cols() + j];
  } else {
    return data()[j * rows() + i];
  }
}

template <typename T>
const T& Matrix<T>::operator()(uint32_t i, uint32_t j) const {
  if (is_row_major()) {
    return data()[i * cols() + j];
  } else {
    return data()[j * rows() + i];
  }
}

template <typename T>
T& Matrix<T>::operator()(uint32_t l) {
  return data_.get()[l];
}

template <typename T>
const T& Matrix<T>::operator()(uint32_t l) const {
  return data_.get()[l];
}

template <typename T>
T& Matrix<T>::at(uint32_t i, uint32_t j) {
  internal::check_bounds(*this, i, j);
  return (*this)(i, j);
}

template <typename T>
const T& Matrix<T>::at(uint32_t i, uint32_t j) const {
  internal::check_bounds(*this, i, j);
  return (*this)(i, j);
}

template <typename T>
T& Matrix<T>::at(uint32_t l) {
  internal::check_bounds(*this, l);
  return (*this)(l);
}

template <typename T>
const T& Matrix<T>::at(uint32_t l) const {
  internal::check_bounds(*this, l);
  return (*this)(l);
}

template <typename T>
template <typename X>
Matrix<T>& Matrix<T>::operator+=(const X& rhs) {
  internal::apply_operator<std::plus<T>>(this, *this, rhs);
  return *this;
}

template <typename T, typename X>
Matrix<T> operator+(Matrix<T> lhs, const X& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T>
template <typename X>
Matrix<T>& Matrix<T>::operator-=(const X& rhs) {
  internal::apply_operator<std::minus<T>>(this, *this, rhs);
  return *this;
}

template <typename T, typename X>
Matrix<T> operator-(Matrix<T> lhs, const X& rhs) {
  lhs -= rhs;
  return lhs;
}

template <typename T>
template <typename X>
Matrix<T>& Matrix<T>::operator*=(const X& rhs) {
  internal::apply_operator<std::multiplies<T>>(this, *this, rhs);
  return *this;
}

template <typename T, typename X>
Matrix<T> operator*(Matrix<T> lhs, const X& rhs) {
  lhs *= rhs;
  return lhs;
}

template <typename T>
template <typename X>
Matrix<T>& Matrix<T>::operator/=(const X& rhs) {
  internal::apply_operator<std::divides<T>>(this, *this, rhs);
  return *this;
}

template <typename T, typename X>
Matrix<T> operator/(Matrix<T> lhs, const X& rhs) {
  lhs /= rhs;
  return lhs;
}

template <typename T, typename X>
Matrix<bool> operator==(const Matrix<T>& lhs, const X& rhs) {
  Matrix<bool> result(lhs.rows(), lhs.cols());
  internal::apply_operator<std::equal_to<T>>(&result, lhs, rhs);
  return result;
}

template <typename T, typename X>
Matrix<bool> operator!=(const Matrix<T>& lhs, const X& rhs) {
  Matrix<bool> result(lhs.rows(), lhs.cols());
  internal::apply_operator<std::not_equal_to<T>>(&result, lhs, rhs);
  return result;
}

template <typename T>
bool Matrix<T>::all() const {
  for (size_t l = 0; l < size(); ++l) {
    if (!data()[l]) return false;
  }
  return true;
}

template <typename T>
bool Matrix<T>::any() const {
  for (size_t l = 0; l < size(); ++l) {
    if (data()[l]) return true;
  }
  return false;
}

template <typename T>
template <typename X>
Matrix<T> Matrix<T>::dot(const Matrix<X>& rhs) const {
  using namespace internal;
  check_dot_dim_match(*this, rhs);

  // Allocate a dst matrix with the same row/col-majorness as this.
  Matrix<T> dst(rows(), rhs.cols(), is_row_major());

  // Invoke the correct style of matrix multiplication based on the
  // row/col-majorness of this.
  if (is_row_major()) {
    if (rhs.is_row_major()) {
      dot_rowwise<ColMajorUnfriendlyIterator>(&dst, *this, rhs);
    } else {
      dot_rowwise<ColMajorFriendlyIterator>(&dst, *this, rhs);
    }
  } else {
    if (rhs.is_row_major()) {
      dot_colwise<RowMajorFriendlyIterator>(&dst, *this, rhs);
    } else {
      dot_colwise<RowMajorUnfriendlyIterator>(&dst, *this, rhs);
    }
  }

  return dst;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() {
  return Matrix(this->data_, this->cols_, this->rows_, !this->is_row_major_);
}

template <typename T>
Matrix<T>::Matrix(std::shared_ptr<T> data, uint32_t rows, uint32_t cols,
                  bool is_row_major)
  : data_(data), rows_(rows), cols_(cols), is_row_major_(is_row_major) {}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T>& m) {
  if (m.is_row_major()) {
    internal::RowMajorFriendlyIterator m_iter(m.rows(), m.cols());
    for (uint32_t maj = 0; maj < m_iter.major_dim_size(); ++maj) {
      for (uint32_t min = 0; min + 1 < m_iter.minor_dim_size(); ++min) {
        stream << m(m_iter.index()) << '\t';
        m_iter.next_minor_dim();
      }
      stream << m(m_iter.index());
      if (maj + 1 < m_iter.major_dim_size()) {
        stream << '\n';
      }
      m_iter.next_major_dim();
    }
  } else {
    internal::RowMajorUnfriendlyIterator m_iter(m.rows(), m.cols());
    for (uint32_t maj = 0; maj < m_iter.major_dim_size(); ++maj) {
      for (uint32_t min = 0; min + 1 < m_iter.minor_dim_size(); ++min) {
        stream << m(m_iter.index()) << '\t';
        m_iter.next_minor_dim();
      }
      stream << m(m_iter.index());
      if (maj + 1 < m_iter.major_dim_size()) {
        stream << '\n';
      }
      m_iter.next_major_dim();
    }
  }
  return stream;
}
