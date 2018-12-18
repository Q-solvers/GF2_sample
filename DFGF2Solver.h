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
#ifndef MPIGF2_DFGF2SOLVER_H
#define MPIGF2_DFGF2SOLVER_H

#include <Eigen/Core>

#include <boost/multi_array.hpp>
#include <alps/numeric/tensors/tensor_base.hpp>

#include <mpi.h>

#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>
#include <queue>
#include <vector>

#include "Timing.h"

#include "DFIntegral.h"
#include "JobsQueue.h"
#include "SimpleGF2Job.h"
#include "GF2Solver.h"

/**
 * @brief GF2Solver class performs self-energy calculation by means of second-order PT using density fitting
 *
 * @author iskakov
 */
class DFGF2Solver : public GF2Solver {
public:
  template<size_t D>
  using ztensor   = alps::numerics::tensor<std::complex<double>, D>;
  template<size_t D>
  using dtensor   = alps::numerics::tensor<double, D>;
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using DFGF2Job  = SimpleGF2Job<3>;
  /**
   * Class constructor
   * Initialize arrays and indices for GF2 loop
   *
   * @param nao -- number of atomic orbitals in cell
   * @param nts -- number of time steps
   * @param symm -- symmetrize selfenergy
   * @param nk -- number of k-points
   * @param NQ -- number of aux basis function in fitted densities
   * @param Gk -- Green's function array defined in (tau,ncell,nao,nao) domain
   * @param Sigma -- Self-energy array
   * @param path -- path to Integrals file
   */
  DFGF2Solver(MPI_Comm comm, int nao, int nts, int nk, int NQ,
            alps::numerics::tensor<std::complex<double>, 4>& Gk, alps::numerics::tensor<std::complex<double>, 4>& Sigma,
              const std::string& path) :
           _comm(comm), _nprocs(0), _myid(0), _nao(nao), _nk(nk), _NQ(NQ), _nts(nts), _path(path),
           Gr_full_tau(Gk),
           Sigma_tau(Sigma),
           _G_k1_tmp(nao,nao),
           _Gb_k2_tmp(nao,nao),
           _G_k3_tmp(nao,nao),
           vijkl(nao, nao, nao, nao),
           vcijkl(nao, nao, nao, nao),
           _queue(comm, nk, nk, nk), 
           statistics(nao*nao*nao*nao*nao*nk*nk*nk*nts){
    MPI_Comm_size(_comm, &_nprocs);
    MPI_Comm_rank(_comm, &_myid);
    init_events();
  }
  /**
   * Solve GF2 equations for Self-energy
   */
  virtual void solve();

private:
  // MPI Communicator to be used
  MPI_Comm _comm;
  // number of CPUs
  int _nprocs;
  // Current CPU id
  int _myid;
  // number of atomic orbitals per cell
  int _nao;
  // number of cells for GF2 loop
  int _nk;
  // auxiliraly basis size
  int _NQ;
  // number of time steps
  int _nts;
  // H5 file with integrals id
  hid_t _file;

  // Path to H5 file
  const std::string _path;

  // references to arrays
  alps::numerics::tensor<std::complex<double>, 4> &Gr_full_tau;
  alps::numerics::tensor<std::complex<double>, 4> & Sigma_tau;

  // Current time step Green's function matrix for k1
  Eigen::MatrixXcd _G_k1_tmp;
  // Current reverse time step Green's function matrix for k2
  Eigen::MatrixXcd _Gb_k2_tmp;
  // Current time step Green's function matrix for k3
  Eigen::MatrixXcd _G_k3_tmp;

  /**
   * Read next part of Coulomb integrals for fixed set of k-points
   */
  void read_next(const std::array<int, 4>& k);

  /**
   * Performs loop over time for fixed set of k-points
   */
  void selfenergy_innerloop(const std::array<int, 4>& k);

  /**
   * Performs all possible contractions for i and n indices
   */
  inline void contraction(int nao2, int nao3,
                          const Eigen::MatrixXcd &G1, const Eigen::MatrixXcd &G2, const Eigen::MatrixXcd &G3,
                          MMatrixXcd & Xm_4,
                          MMatrixXcd & Xm_1, MMatrixXcd & Xm_2,
                          MMatrixXcd & Ym_1, MMatrixXcd & Ym_2,
                          const MMatrixXcd &vm_1, DFGF2Solver::MMatrixXcd &Xm, MMatrixXcd &Vm, MMatrixXcd &Sm);


  /**
   * Compute two-electron integrals for the fixed set of k-points using pre-computed fitted densities
   *
   * @param set of k-points
   */
  void setup_integrals(const std::array<int, 4>& k);




  // Pre-computed fitted densities
  DFIntegral *_coul_int1;
  DFIntegral *_coul_int2;
  DFIntegral *_coul_int5;
  DFIntegral *_coul_int6;

  ztensor<4> vijkl;
  ztensor<4> vcijkl;

  /**
   * Computes next set of indices
   * In case of MPI parallelization, _nprocs-1 CPU will work as a master and distribute workload to the rest CPUs.
   */
  JobsQueue<DFGF2Job> _queue;

  //
  ExecutionStatistic statistics;

  void init_events() {
    statistics.add("total");
    statistics.add("read");
    statistics.add("setup");
    statistics.add("nao");
    statistics.add("reduce");
  }
};


#endif //MPIGF2_DFGF2SOLVER_H
