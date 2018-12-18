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

#include <iostream>

#include <mpi.h>
#include <alps/hdf5.hpp>
#include <alps/numeric/tensors/tensor_base.hpp>
#include <alps/params.hpp>
#include "DFGF2Solver.h"

void define_parameters(alps::params &p) {
  p.description("PBC GF2 Solver");
  p.define<std::string>("input_file", "input.h5" ,"Input file with an initial Green's function").
    define<std::string>("output_file", "output.h5" ,"Output file with a resulting Self-energy").
    define<std::string>("dfintegral_file", "df_int.h5" ,"HDF5 file with density fitted Coulomb integrals");
}

template<size_t Dim>
using ztensor = alps::numerics::tensor<std::complex<double>, Dim >;
template<size_t Dim>
using dtensor = alps::numerics::tensor<double, Dim >;
template<size_t Dim>
using itensor = alps::numerics::tensor<int, Dim >;

/*! \mainpage
 *
 * \section intro Introduction
 *
 * \section instal_sec Installation
 *
 * \section df Density fitting
 *
 */

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  alps::mpi::communicator comm;
  alps::params p(argc, argv);
  define_parameters(p);
  if(p.help_requested()) {
    if(!comm.rank()) {
      p.print_help(std::cout);
    }
    return 0;
  } else {
    if(!comm.rank()) {
      std::cout << p;
    }
  }
  
  ztensor<4> G_tau;
  alps::hdf5::archive ar(p["input_file"].as<std::string>(), "r");
  ar["G_tau/data"] >> G_tau;
  ar.close();
  
  ztensor<4> Sigma(G_tau.shape());
  int nts = Sigma.shape()[0];
  int nk = Sigma.shape()[1];
  int nao = Sigma.shape()[2];
  
  int NQ = 0;
  {
    ar.open(p["dfintegral_file"].as<std::string>(), "r");
    dtensor<4> df_int;
    ar["VQ/0"]>>df_int;
    NQ = df_int.shape()[1];
    ar.close();
  }
  std::cout<<NQ<<std::endl;
  
  DFGF2Solver solver(comm, nao, nts, nk, NQ, G_tau, Sigma, p["dfintegral_file"].as<std::string>());
  solver.solve();
  if(!comm.rank()) {
    ar.open(p["output_file"].as<std::string>(), "r");
    ztensor<4> Sigma_true;
    ar["Selfenergy/data"]>>Sigma_true;
    ar.close();
    Sigma_true -= Sigma;
    double norm = std::abs(*std::max_element(Sigma_true.data(), Sigma_true.num_elements() + Sigma_true.data(),
                                            [](std::complex<double> a, std::complex<double> b) {return std::abs(a) < std::abs(b);} ) );
    if ( norm < 1e-12 ) {
      std::cout<<"results are identical. norm_2 of difference is "<<std::scientific<<norm<<std::endl;
    } else {
      std::cerr<<"results are different. norm_2 of difference is "<<std::scientific<<norm<<std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}