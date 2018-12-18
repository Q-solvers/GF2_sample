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
#ifndef GF2_JOBSQUEUE_H
#define GF2_JOBSQUEUE_H

#include <mpi.h>
#include <vector>
#include <queue>
#include <numeric>
#include <iostream>
#include <array>


/**
 * @brief Decompose flat index into indices set based on the size of each dimension
 */
template<size_t S, int D>
struct indexer{
  static void idx(std::array<int, S> & inds, const std::array<int, S> & dim_array, int ind) {
    inds[D - 1] = ind % dim_array[D - 1];
    indexer<S, D-1>::idx(inds, dim_array, ind/dim_array[D - 1]);
  }
};

/// Partial specification for one-dimensional case
template<size_t S>
struct indexer<S, 1>{
  static void idx(std::array<int, S> & inds, const std::array<int, S> & dim_array, int ind) {
    inds[0] = ind;
  }
};

/**
 * @brief This class will implement a queue for the workload distribution.
 * It will have a queue-object with atomic jobs. Each job consists of set of indices
 * that need to be processed in self-energy evaluation.
 *
 * @tparam JobPieceType
 */
template<typename JobPieceType>
class JobsQueue {
  static constexpr int SIZE = JobPieceType::SIZE;

public:

  template<class...dim_types>
  JobsQueue(MPI_Comm comm, dim_types...dims) : _nprocs(1), _myid(0), _comm(comm), _dim_array{{dims...}} {
    static_assert(sizeof...(dims) == SIZE, "Wrong number of dimensions");
    if(comm != MPI_COMM_NULL) {
      MPI_Comm_size(comm, &_nprocs);
      MPI_Comm_rank(comm, &_myid);
    }
    reset(dims...);
  }

  /**
   * Fill queue with initial values, based on the sizes of all dimensions
   * @tparam dim_types
   * @param dims - sizes of each dimension
   */
  template<class...dim_types>
  void reset(dim_types...dims) {
    static_assert(sizeof...(dims) == SIZE, "Wrong number of dimensions");
    if(_myid != (_nprocs - 1) ) {
      return;
    }
    int worker_number = (_nprocs == 1 ? 1 : _nprocs - 1);
    std::queue<JobPieceType> empty;
    std::swap( _jobsQueue, empty );
    _total = std::accumulate(_dim_array.begin(), _dim_array.end(), 1, std::multiplies<int>());
    int local_total = _total / worker_number;
    for(int i = 0; i < worker_number; ++i) {
      std::array<int, 2> inds;
      inds[0] =     i * local_total;
      inds[1] = (i+1) * local_total;
      JobPieceType pieceType = JobPieceType(inds);
      _jobsQueue.push(pieceType);
    }
    for(int i = local_total * worker_number; i < _total; ++i) {
      std::array<int, 2> inds;
      inds[0] = i;
      inds[1] = i+1;
      JobPieceType pieceType = JobPieceType(inds);
      _jobsQueue.push(pieceType);
    }
    std::cout<<"Inited jobs queue with "<<_jobsQueue.size()<<" atomic jobs"<<std::endl;
  }

  std::array<int, SIZE> get_indices(int i) const {
    std::array<int, SIZE> inds;
    indexer<SIZE, SIZE>::idx(inds, _dim_array, i);
    return inds;
  };

  /**
   * Get next available job portion. If there is nothing left in the queue the 'nothing-to-do' job will be returned
   *
   * @return job portion
   */
  JobPieceType next()  {
    // we have single CPU, run simple job
    if(_nprocs == 1) {
      return single();
    } else {
      return master_slave();
    }
  }

  /**
   * Get access to underlying std::queue object for test purpose
   *
   * @return std::queue object
   */
  const std::queue<JobPieceType> &jobsQueue() const {
    return _jobsQueue;
  }

private:

  // MPI part
  /// total number of MPI-processes
  int _nprocs;
  /// MPI-process Id
  int _myid;
  /// MPI communicator
  MPI_Comm _comm;
  
    /// job queue
  std::queue<JobPieceType> _jobsQueue;
  /// total size
  int _total;
  /// dimenstions
  std::array<int, SIZE> _dim_array;

  JobPieceType single() {
    JobPieceType NOTHING_TO_DO(-1);
    // check if there is any job quant
    if(!_jobsQueue.empty()) {
      JobPieceType front = _jobsQueue.front();
      _jobsQueue.pop();
      progress(front);
      return front;
    } else {
      // return nothing-to-do job
      return NOTHING_TO_DO;
    }
  }

  JobPieceType master_slave() {
    MPI_Status status;
    JobPieceType NOTHING_TO_DO(-1);
    JobPieceType N(-1);
    if((_myid == _nprocs - 1) && !_jobsQueue.empty()) {
      // master process
      // work termination flag
      // temporary variable for communication initialization
      int k;
      // number of CPUs that have finished their job
      int term = 0;
      // run loop while number of cpus that finished their job is less than total number of cpus
      // or total number of points.
      while(term < _nprocs - 1) {
        // get initiation message from
        MPI_Recv(&k, 1, MPI_INT, MPI_ANY_SOURCE, 0, _comm, &status);
        // check that we have point to send
        if(!_jobsQueue.empty()) {
          // get next point
          JobPieceType j = _jobsQueue.front();
          _jobsQueue.pop();
          progress(j);
          // send next point to compute
          MPI_Send(j.rawdata(), j.size(), MPI_INT, status.MPI_SOURCE, 0, _comm);
        } else {
          // there are no more points left
          MPI_Send(NOTHING_TO_DO.rawdata(), NOTHING_TO_DO.size(), MPI_INT, status.MPI_SOURCE, 0, _comm);
          // increase number of cpus that finished their job
          ++term;
        }
      }
      /// All jobs have been finished, master process have nothing to do
      return NOTHING_TO_DO;
    } else {
      // slave process
      // send initiation message
      int k = 0;
      MPI_Send(&k, 1, MPI_INT, _nprocs - 1, 0, _comm);
      // request next point
      MPI_Recv(N.rawdata(), N.size(), MPI_INT, _nprocs - 1, 0, _comm, &status);
      return N;
    }
  }

  void progress(const JobPieceType &j) const {
    int progress = _total - _jobsQueue.size() - 1;
    if(progress % (_total / 50) == 0 ) {
      std::cout << (100*progress)/_total << "% done. Currently processing job: " << j << std::endl;
    }
  }
};


#endif //GF2_JOBSQUEUE_H
