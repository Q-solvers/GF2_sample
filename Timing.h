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
#ifndef GF2_TIMING_H
#define GF2_TIMING_H

#include <mpi.h>
#include <string>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief ExecutionStatistic class
 *
 * @author iskakoff
 */
    class ExecutionStatistic {
    public:

      //ExecutionStatistic(): _system_size(1) {}
      
      ExecutionStatistic(int size): _system_size(size) {}

      void add(const std::string& name) {
        if(_events.find(name) == _events.end()) {
          _events[name] = std::make_pair(0.0, 0.0);
        }
      }

      /**
       * Update event time
       * @param name - event name
       */
      void end(const std::string& name) {
//#ifdef _OPENMP
//        if(omp_get_thread_num()) {
//          return;
//        }
//#endif
        double time1 = time();
        _events[name] = std::make_pair(_events[name].first + time1 - _events[name].second, time1);
      }

      /**
       * register the start point of the event
       *
       * @param name - event name
       */
      void start(const std::string& name) {
//#ifdef _OPENMP
//        if(omp_get_thread_num()) {
//          return;
//        }
//#endif
        _events[name] = std::make_pair(_events[name].first, time());
      }

      /**
       * Print all observed events
       */
      void print() {
        std::cout<<"Execution statistics:"<<std::endl;
        for (auto& kv : _events) {
          std::cout <<"Event "<< kv.first << " took " << kv.second.first << " s.; "<< _system_size/kv.second.first<<" nop/s" << std::endl;
        }
        std::cout<<"====================="<<std::endl;
      }

      void print(MPI_Comm comm) {
        int id, np;
        MPI_Comm_rank(comm, &id);
        MPI_Comm_size(comm, &np);

        std::vector<double> max(_events.size(), 0.0);
        int i = 0;
        for (auto& kv : _events) {
          max[i] = kv.second.first;
          ++i;
        }
        if(!id) {
          MPI_Reduce(MPI_IN_PLACE, max.data(), max.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
        } else {
          MPI_Reduce(max.data(), max.data(), max.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
        }
        if(!id) {
          std::cout << "Execution statistics: "<< std::endl;
          i = 0;
          for (auto &kv : _events) {
            std::cout << "Event " << kv.first << " took "<<max[i]<<" s.; "<<_system_size/max[i]<<" nop/s"<< std::endl;
            ++i;
          }
          std::cout << "=====================" << std::endl;
        }
      }

      /**
       * Return event timing pair
       * @param event_name - event name
       * @return event timing
       */
      std::pair<double, double> event(const std::string & event_name) {
        if(_events.find(event_name) != _events.end()) {
          return _events[event_name];
        }
        return std::make_pair(0.0, 0.0);
      };
    private:
      // registered events timing pairs
      // pair.first corresponds to total event time
      // pair.second corresponds to last time when event was happened
      std::map<std::string, std::pair<double, double> > _events;
      
      int _system_size;

      double time() const {
        return MPI_Wtime();
      }
    };

#endif //GF2_TIMING_H