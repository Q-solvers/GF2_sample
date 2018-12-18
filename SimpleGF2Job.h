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
#ifndef GF2_SIMPLEDFGF2JOB_H
#define GF2_SIMPLEDFGF2JOB_H


#include <array>
#include <algorithm>

/**
 * @brief SimpleGF2Job class defines job portion for self-energy evaluation.
 *
 * Simple multi-dimensional GF2 job. Each job's dimension corresponds to the index used in MPI-parallelization.
 * We will fill queue object with jobs, that contains set of indices for current part of the self-energy evaluation.
 *
 * @tparam D - job dimension
 */
template<int D = 2>
class SimpleGF2Job {
public:
  /// Job dimension
  static constexpr int SIZE = D;
  /**
   * Fill job indices with specified value
   *
   * @param value - value for all job indices
   */
  SimpleGF2Job(int value) {
    std::fill(_data.begin(), _data.end(), value);
  }

  /**
   * Construct job from array
   *
   * @param data - indices array
   */
  SimpleGF2Job(const std::array<int, 2> & data) : _data(data) {}
  SimpleGF2Job(std::array<int, 2> && data) : _data(data) {}

  /**
   * @return job indices array
   */
  const std::array<int, 2>& data() const {
    return _data;
  }

  /**
   * Pointer to the underlying data structure. We need to have this raw pointer to pass Job over MPI send-receive.
   *
   * @return pointer to the underlying data structure
   */
  int* rawdata() {
    return _data.data();
  }
  /**
   * @return size of job
   */
  size_t size() const {
    return _data.size();
  }

  /**
   * Check job for validness. We assume that job is valid when all its indices are not negative
   *
   * @return true if job is valid
   */
  bool valid() {
    return (*std::min_element(_data.begin(), _data.end()) ) >= 0;
  }

  /**
   * Print job into std stream
   *
   * @param stream - output stream
   * @param job    - job to print
   */
  friend std::ostream & operator <<(std::ostream& stream, const SimpleGF2Job<D> & job) {
    for(int i = 0; i<D; ++i)stream<<(i==0? "":" ")<<job._data[i];
    return stream;
  }

private:
  // min and max index to process within the job
  std::array<int, 2> _data;
};


#endif //GF2_SIMPLEDFGF2JOB_H
