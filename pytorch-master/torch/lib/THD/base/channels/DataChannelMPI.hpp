#pragma once

#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include <mpi.h>
#include <memory>
#include <utility>
#include <unordered_map>
#include <vector>

namespace thd {

struct DataChannelMPI : DataChannel {
  struct RequestMPI : DataChannel::Request {
    friend class DataChannelMPI; // allows `DataChannelMPI` to access private members

    RequestMPI();
    virtual ~RequestMPI();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    template<typename T>
    void steal_buffer(std::shared_ptr<T> ptr);
    MPI_Request& new_request();

    std::vector<std::shared_ptr<void>> _buffers;
    std::vector<MPI_Request> _requests;
  };

  DataChannelMPI();
  virtual ~DataChannelMPI();

  bool init() override;

  int getRank() override;
  int getNumProcesses() override;

  void allGather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
                 THDGroup group_id = THDGroupWORLD) override;
  void gather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
              int dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void scatter(std::vector<thpp::Tensor*>& input, thpp::Tensor& output,
               int src_rank, THDGroup group_id = THDGroupWORLD) override;
  void allReduce(thpp::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD) override;
  void reduce(thpp::Tensor& data, THDReduceOp operation, int dst_rank,
              THDGroup group_id = THDGroupWORLD) override;
  void broadcast(thpp::Tensor& data, int src_rank,
                 THDGroup group_id = THDGroupWORLD) override;
  void send(const Scalar& data, int dst_rank) override;
  void send(thpp::Tensor& data, int dst_rank) override;
  void receive(Scalar& data, int src_rank) override;
  void receive(thpp::Tensor& data) override;
  void receive(thpp::Tensor& data, int src_rank) override;
  RequestMPI* isend(thpp::Tensor& data, int dst_rank) override;
  RequestMPI* ireceive(thpp::Tensor& data, int src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;
  THDGroup newGroup(const std::vector<int>& ranks) override;

private:
  void _broadcastPack(thpp::Tensor& data, int src_rank, MPI_Comm comm) const;
  void _broadcastUnpack(thpp::Tensor& data, int src_rank, MPI_Comm comm) const;

  int _rank; // Current process' rank
  int _num_processes; // Number of processes in network

  // Existing groups of processes with assigned MPI communicator
  // and corresponding group ids
  std::unordered_map<THDGroup, std::pair<MPI_Comm, DataChannel::Group>> _groups;
};

} // namespace thd
