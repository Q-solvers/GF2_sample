#include "DFGF2Solver.h"

void DFGF2Solver::selfenergy_innerloop(const std::array<int, 4>& k) {
  statistics.start("nao");
  int nao2 = _nao * _nao;
  int nao3 = _nao * _nao * _nao;
  int momshift = k[0] * nao2;
#pragma omp parallel
  {
  Eigen::MatrixXcd G1(_nao, _nao);
  Eigen::MatrixXcd G2(_nao, _nao);
  Eigen::MatrixXcd G3(_nao, _nao);
    ztensor<3> X(_nao, _nao, _nao);
    MMatrixXcd Xm_4(X.data(), _nao, nao2);
    ztensor<3> X1(_nao, _nao, _nao);
    MMatrixXcd Xm_1(X1.data(), nao2, _nao);
    MMatrixXcd Xm_2(X1.data(), _nao, nao2);
    ztensor<3> Y1(_nao, _nao, _nao);
    MMatrixXcd Ym_1(Y1.data(), nao2, _nao);
    MMatrixXcd Ym_2(Y1.data(), _nao, nao2);
    MMatrixXcd Xm(X.data(), 1, nao3);
    MMatrixXcd Vm(vcijkl.data(), nao3, _nao);

    // Loop over tau indices
#pragma omp for
  for(int t = 0; t< _nts;++t) {
    int shift = t * _nk * nao2 + momshift;
    int tt = _nts - t - 1;
    // initialize Green's functions
    for (int q0 = 0; q0 < _nao; ++q0) {
      for(int p0 = 0; p0 < _nao; ++p0) {
        G1(q0, p0)  = Gr_full_tau(t , k[1], q0, p0);
        G2(q0, p0) = Gr_full_tau(tt, k[2], q0, p0);
        G3(q0, p0)  = Gr_full_tau(t , k[3], q0, p0);
      }
    }
    for(int i = 0; i < _nao; ++i) {
      // pm,k
      MMatrixXcd Sm(Sigma_tau.data() + shift + i * _nao, 1, _nao);
      MMatrixXcd vm_1(vijkl.data() + i * nao3, nao2, _nao);
      contraction(nao2, nao3, G1, G2, G3, Xm_4, Xm_1, Xm_2, Ym_1, Ym_2, vm_1, Xm, Vm, Sm);
    }
  }
  }
  statistics.end("nao");
}

void DFGF2Solver::contraction(int nao2, int nao3,
                              const Eigen::MatrixXcd &G1, const Eigen::MatrixXcd &G2, const Eigen::MatrixXcd &G3,
                              MMatrixXcd & Xm_4,
                              MMatrixXcd & Xm_1, MMatrixXcd & Xm_2,
                              MMatrixXcd & Ym_1, MMatrixXcd & Ym_2,
                              const MMatrixXcd &vm_1, DFGF2Solver::MMatrixXcd &Xm, MMatrixXcd &Vm, MMatrixXcd &Sm) {
  // pm,l = pm,k k,l
  Xm_1.noalias() = (vm_1 * G3);
  // ml,q = ml,p p,q
  Ym_1.noalias() = (Xm_2.transpose() * G1);
  // n,lq = n,m m,lq
  Xm_2.noalias() = G2 * Ym_2;
  // q,nl
  Xm_4.noalias() = Xm_1.transpose();
  // i,j = i,qnl qnl,j
  Sm.noalias() += Xm * Vm;
//  Sigma_tau(t, k_mom[0]).matrix() += Xm * Vm;
}
