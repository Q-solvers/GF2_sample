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
#ifndef GF2_DFINTEGRAL_H
#define GF2_DFINTEGRAL_H


#include <hdf5.h>
#include <hdf5_hl.h>

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include "base_integral.h"


/**
 * @brief Integral class read Density fitted 3-center integrals from HDF5 file
 */
class DFIntegral : public base_integral {
// prefixes for hdf5
  const std::string rval_ = "VQ";
  const std::string ival_ = "ImVQ";
public:
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  template<size_t dim>
  using ztensor = alps::numerics::tensor<std::complex<double>, dim>;
  template<size_t dim>
  using dtensor = alps::numerics::tensor<double, dim>;
  DFIntegral(const std::string & path, int nao, int nk, int NQ, IntegralType type = first_integral) : base_integral(path, nk, type), _dim(nk),
                                      _k1(-1), _k2(-1), _chunk_size(0), _current_chunk(-1),_vij_Q(1, NQ, nao, nao) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    H5LTread_dataset_int(file,"chunk_size",&_chunk_size);
    H5Fclose(file);
    _vij_Q.reshape(_chunk_size, NQ, nao, nao);
  }


  virtual ~DFIntegral() {}

  /**
   * read next part of the interaction integral for the leading index 'n'
   * @param file - file to be used
   * @param n - leading index
   */
  void read_integrals(hid_t file, int k1, int k2, IntegralType type = first_integral){
    // avoid unnecessary reading
//    if(k1==_k1 && k2==_k2) {
//      return;
//    }
    int flat_index = k1 * _dim + k2;
    _k1 = k1;
    _k2 = k2;
    if( (flat_index/_chunk_size) == _current_chunk) {
      // we have data cached
      return;
    }
    _current_chunk = flat_index/_chunk_size;

    // Construct integral dataset name
    std::string inner = "/" + std::to_string(_current_chunk * _chunk_size);
    std::string dsetnumr = rval_ + inner;
    std::string dsetnumi = ival_ + inner;
    // read data
    H5LTread_dataset_double(file,dsetnumr.c_str(), reinterpret_cast<double *>(_vij_Q.data()));
  };


  const ztensor<4> &vij_Q() const {
    return _vij_Q;
  }

  int wrap(int k1, int k2) const {
    return (k1 * _dim + k2) % _chunk_size;
  }


private:

  int _dim;
  // current leading index
  int _k1;
  int _k2;
  int _chunk_size;
  int _current_chunk;
  
    // Coulomb integrals stored in density fitting format
  ztensor<4> _vij_Q;

};


#endif //GF2_DFINTEGRAL_H
