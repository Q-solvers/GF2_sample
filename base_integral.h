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
#ifndef GF2_BASE_INTEGRAL_H
#define GF2_BASE_INTEGRAL_H

#include <hdf5.h>
#include <alps/numeric/tensors.hpp>

enum IntegralType{first_integral, second_direct, second_exchange};

class base_integral {
public:
  base_integral(const std::string & path, int nk, IntegralType type) : _mom_cons(nk,nk,nk), _type(type) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    alps::numerics::tensor<double, 2> kmesh(nk, 3);
    H5LTread_dataset_double(file,"grid/k_mesh_scaled",kmesh.data());
    H5Fclose(file);
    for (int i = 0; i < nk; ++i) {
      alps::numerics::tensor<double, 1> ki = kmesh(i);
      for (int j = 0; j < nk; ++j) {
        alps::numerics::tensor<double, 1> kj = kmesh(j);
        for (int k = 0; k < nk; ++k) {
          alps::numerics::tensor<double, 1> kk = kmesh(k);
          alps::numerics::tensor<double, 1> kl = wrap(kk + ki - kj);
          _mom_cons(i, j ,k) = find_pos(kl, kmesh);
        }
      }
    }
  }

  virtual ~base_integral(){}

  /**
   * compute integral momenta using momentum conservation of the first integral
   *
   * @param n - first integral momentum triplet
   * @return full 4 momenta for the current triplet
   */
  std::array<int, 4> momentum_conservation(const std::array<int, 3> &n) const {
    std::array<int,4> k;
    if(_type == first_integral) {
      k[0] = n[0];
      k[1] = n[1];
      k[2] = n[2];
      k[3] = _mom_cons(n[0], n[1], n[2]);
    } else if(_type == second_direct) {
      k[0] = n[1];
      k[1] = n[0];
      k[2] = _mom_cons(n[0], n[1], n[2]);
      k[3] = n[2];
    } else {
      k[0] = _mom_cons(n[0], n[1], n[2]);
      k[1] = n[0];
      k[2] = n[1];
      k[3] = n[2];
    }
    return k;
  };

  const IntegralType &type() const {
    return _type;
  }

private:
  // Momentum conservation tensor
  alps::numerics::tensor<int, 3> _mom_cons;
  IntegralType _type;

  alps::numerics::tensor<double, 1> wrap(const alps::numerics::tensor<double, 1> & k) {
    alps::numerics::tensor<double, 1> kk = k;
    for (int j = 0; j < kk.shape()[0]; ++j) {
      while( (kk(j) - 9.9999999999e-1) > 0.0) {
        kk(j) -= 1.0;
      }
      if(std::abs(kk(j)) < 1e-9) {
        kk(j) = 0.0;
      }
      while(kk(j) < 0) {
        kk(j) += 1.0;
      }
    }
    return kk;
  };

  int find_pos(const alps::numerics::tensor<double, 1> & k, const alps::numerics::tensor<double, 2> & kmesh) {
    for (int i = 0; i < kmesh.shape()[0]; ++i) {
      bool found = true;
      for (int j = 0; j < k.shape()[0]; ++j) {
        found &= std::abs(k(j) - kmesh(i, j)) < 1e-12;
      }
      if(found) {
        return i;
      }
    }
    throw std::logic_error("K point (" + std::to_string(k(0)) + ", " + std::to_string(k(1))  + ", " + std::to_string(k(2)) + ") has not been found in the mesh.");
  }
};


#endif //GF2_BASE_INTEGRAL_H
