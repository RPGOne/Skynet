#include "../base/channels/DataChannelTCP.hpp"
#include "TestUtils.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <thread>

constexpr int WORKERS_NUM = 2;
constexpr int MASTER_PORT = 45680;

void master()
{
  setenv("WORLD_SIZE", std::to_string((WORKERS_NUM + 1)).data(), 1);
  setenv("RANK", "0", 1);
  setenv("MASTER_PORT", std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelTCP>(2000); // timeout after 2s

  ASSERT_THROWS(std::exception, masterChannel->init())
}


int main() {
  std::thread master_thread(master);
  master_thread.join();
  std::cout << "OK" << std::endl;
  return 0;
}
