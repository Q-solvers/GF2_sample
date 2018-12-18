/*
 * Copyright (c) 2018 Sergei Iskakov.
 *
 * This program is free software: you can redistribute it and/or modify..
 * it under the terms of the GNU General Public License as published by..
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful, but.
 * WITHOUT ANY WARRANTY; without even the implied warranty of.
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU.
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License.
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MPIGF2_GF2SOLVER_H
#define MPIGF2_GF2SOLVER_H

#include <alps/numeric/tensors.hpp>

class GF2Solver {
public:
  template<size_t D>
  using ztensor = alps::numerics::tensor<std::complex<double>, D>;

  virtual void solve() = 0;

  virtual ~GF2Solver() {}

};


#endif //MPIGF2_GF2SOLVER_H
