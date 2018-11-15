#include <iostream>

#include "matrix.h"

int main() {
  Matrix<double> m(2, 3, {1., 2., 3., 4., 5., 6.});
  Matrix<float> m2(2, 3, {6., 5, 4, 3, 2, 1});
  std::cout << "m =\n" << m << std::endl;
  std::cout << "m.T =\n" << m.transpose() << std::endl;
  std::cout << "m2 =\n" << m2 << std::endl;
  std::cout << "m + m2 =\n" << (m + m2) << std::endl;

  m2 += m;
  std::cout << "m2 =\n" << m2 << std::endl;
  m2 /= m;
  std::cout << "m2 =\n" << m2 << std::endl;

  std::cout << "m2 == m =\n" << (m2 == m) << std::endl;
  std::cout << "m2 == m2 =\n" << (m2 == m2) << std::endl;

  Matrix<double> m3(1, 3, {8., 9, 10});
  Matrix<double> m4(2, 1, {8., 9});

  std::cout << "m + m3 =\n" << (m + m3) << std::endl;
  std::cout << "m + m4 =\n" << (m + m4) << std::endl;
  std::cout << "m + 1 =\n" << (m + 1) << std::endl;

  std::cout << "m.transpose() =\n" << m.transpose() << std::endl;
  std::cout << "m.transpose() + m4.transpose() =\n"
            << (m + m4).transpose() << std::endl;

  return 0;
}
