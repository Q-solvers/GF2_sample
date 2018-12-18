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
#include "DFGF2Solver.h"
#include <boost/lexical_cast.hpp>
#include <hdf5_hl.h>
#include <Eigen/Sparse>

#include <fstream>

void DFGF2Solver::solve(/*Integrals & integrals*/) {
  //
  _coul_int1 = new DFIntegral(_path, _nao, _nk, _NQ);
  _coul_int2 = new DFIntegral(_path, _nao, _nk, _NQ);
  _coul_int5 = new DFIntegral(_path, _nao, _nk, _NQ, second_exchange);
  _coul_int6 = new DFIntegral(_path, _nao, _nk, _NQ, second_exchange);
  // open file with integrals` data
  _file = H5Fopen(_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  // clean self_energy array
  Sigma_tau.set_zero();
  DFGF2Job N(-1);
  _queue.reset(_nk, _nk, _nk);
  statistics.start("total");
  // start main loop
  // execution will proceed while current point is non-negative
  while((N = _queue.next()).valid()) {
    for(int k1k3k2 = N.data()[0]; k1k3k2 < N.data()[1]; ++k1k3k2) {
      int k1 = k1k3k2 / (_nk * _nk);
      int k3 = (k1k3k2 / _nk) % _nk;
      int k2 = k1k3k2 % _nk;
//      for (int k2 = 0; k2 < _nk; ++k2) {
        std::array<int, 4> k = _coul_int1->momentum_conservation({{k1, k2, k3}});
        statistics.start("read");
        // read next part of integrals
        read_next(k);
        statistics.end("read");
        statistics.start("setup");
        setup_integrals(k);
        statistics.end("setup");
        selfenergy_innerloop(k);
          // print execution time
//      }
    }
  }
  statistics.start("reduce");
  // collect data from neighbors
  MPI_Allreduce(MPI_IN_PLACE, Sigma_tau.data(), Sigma_tau.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, _comm);
  statistics.end("reduce");
  // Normalize Self-energy
  Sigma_tau /= (_nk*_nk);
  statistics.end("total");
  // print execution time
  statistics.print(_comm);
  // close file and exit
  H5Fclose(_file);
  delete _coul_int1;
  delete _coul_int2;
  delete _coul_int5;
  delete _coul_int6;
}

void DFGF2Solver::read_next(const std::array<int, 4>& k) {
  _coul_int1->read_integrals(_file, k[0], k[1]);
  _coul_int2->read_integrals(_file, k[2], k[3]);
  _coul_int5->read_integrals(_file, k[0], k[3]);
  _coul_int6->read_integrals(_file, k[2], k[1]);
}

void DFGF2Solver::setup_integrals(const std::array<int, 4>& kpts) {
  vijkl.set_zero();
  vcijkl.set_zero();
  int k1 = _coul_int1->wrap(kpts[0], kpts[1]);
  int k2 = _coul_int2->wrap(kpts[2], kpts[3]);
  int k3 = _coul_int5->wrap(kpts[0], kpts[3]);
  int k4 = _coul_int6->wrap(kpts[2], kpts[1]);
  CMMatrixXcd vx1(_coul_int5->vij_Q().data() + k3 * _nao *_nao *_NQ, _NQ, _nao * _nao);
  CMMatrixXcd vx2(_coul_int6->vij_Q().data() + k4 * _nao *_nao *_NQ, _NQ, _nao * _nao);
  CMMatrixXcd v1(_coul_int1->vij_Q().data() + k1 * _nao *_nao *_NQ, _NQ, _nao * _nao);
  CMMatrixXcd v2(_coul_int2->vij_Q().data() + k2 * _nao *_nao *_NQ, _NQ, _nao * _nao);
  MMatrixXcd v(vijkl.data(), _nao*_nao, _nao*_nao);
  MMatrixXcd vx(vijkl.data(), _nao*_nao, _nao*_nao);
  MMatrixXcd vc(vijkl.data(), _nao*_nao, _nao*_nao);

  vc = 2.0*v1.transpose().conjugate() * v2.conjugate();

#pragma omp parallel for
  for (int i = 0; i < _nao; ++i) {
    for (int j = 0; j < _nao; ++j) {
      for (int k = 0; k < _nao; ++k) {
        for (int l = 0; l < _nao; ++l) {
          vcijkl(j,k,l,i) = vijkl(i, j, k, l);
        }
      }
    }
  }
  vx = vx1.transpose().conjugate() * vx2.conjugate();
#pragma omp parallel for
  for (int i = 0; i < _nao; ++i) {
    for (int j = 0; j < _nao; ++j) {
      for (int k = 0; k < _nao; ++k) {
        for (int l = 0; l < _nao; ++l) {
          vcijkl(j,k,l,i) -= vijkl(i, l, k, j);
        }
      }
    }
  }
  v = v1.transpose() * v2;
}
