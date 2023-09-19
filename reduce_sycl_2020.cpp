/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Simple reductions
 *
 *  @filename reduce_sycl_2020
 *
 **************************************************************************/

#include <CL/sycl.hpp>
#include <numeric>
#include <vector>
#include <chrono>
#include <iostream>
// SYCL atomic reference
template <typename T>
cl::sycl::event call_reduce_func0(sycl::queue &q, T *input, T *output, size_t size) {
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> id) {
      sycl::atomic_ref<T, sycl::memory_order::relaxed,
                       sycl::memory_scope::system,
                       sycl::access::address_space::global_space>(output[0]) +=
          input[id];
    });
  });
}

// SYCL reduce Function
template <typename T>
cl::sycl::event call_reduce_func1(sycl::queue &q, T *input, T *output, size_t size) {

  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{size},
                     sycl::reduction(output, sycl::plus<>()),
                     [=](sycl::id<1> id, auto &sum) { sum += input[id]; });
  });
}

// Reduce over group
template <typename T>
cl::sycl::event call_reduce_func2(sycl::queue &q, T *input, T *output, size_t size) {

  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(size), sycl::range<1>(256)),
        [=](sycl::nd_item<1> id) {
          size_t chunk = size / (id.get_group_range()[0]);
          T *start = input + (((size_t)id.get_group()[0]) * chunk);
          T sum = reduce_over_group(
              id.get_group(), start[id.get_local_id(0)], sycl::plus<>());
          if (id.get_local_id(0) == 0) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space>(
                output[0]) += sum;
          }
        });
  });
}



// SYCL reduce over group joint reduce
template <typename T>
cl::sycl::event call_reduce_func3(sycl::queue &q, T *input, T *output, size_t size) {

  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(size), sycl::range<1>(256)),
        [=](sycl::nd_item<1> id) {
          size_t chunk = size / (id.get_group_range()[0]);
          auto start = input + (((size_t)id.get_group()[0]) * chunk);
          auto end = start + chunk;
          T sum = joint_reduce(id.get_group(), start, end, sycl::plus<>());
          if (id.get_local_id(0) == 0) {
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::system,
                             sycl::access::address_space::global_space>(
                output[0]) += sum;
          }
        });
  });
}

int main(int argc, char *argv[]) {
  using type_t = double; 
  int alg = 0;
  if (argc > 1)
    alg = atoi(argv[1]);
  size_t iter = 2;
  if(argc >2)
    iter = atoi(argv[2]);
  size_t size = 1 << 15;
  if (argc > 3)
    size = atoi(argv[2]);
  // Use inorder queue
sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::enable_profiling()}};
std::vector<type_t> input(size);
std::iota(input.begin(), input.end(), 0);
type_t output = 0.0f;
auto d_input = sycl::malloc_device<type_t>(size, q);
auto d_output = sycl::malloc_device<type_t>(1, q);
cl::sycl::event ev;
q.copy(input.data(), d_input, size, std::vector<sycl::event>());

q.copy(&output, d_output, 1, std::vector<sycl::event>());
q.wait();
uint64_t average_elapsed_ns {0};
std::chrono::duration<double> overall {0};
for(int i=0; i<iter; i++) {
   const auto start = std::chrono::steady_clock::now();
  switch (alg) {
  case 0:
    ev = call_reduce_func0(q, d_input, d_output, size);
    break;
  case 1:
    ev = call_reduce_func1(q, d_input, d_output, size);
    break;
  case 2:
    ev = call_reduce_func2(q, d_input, d_output, size);
    break;
  case 3:
    ev = call_reduce_func3(q, d_input, d_output, size);
    break;

  default:
    throw std::overflow_error(
        "unknown algorithm please choose a number between 0-3.");

    break;
  }
  q.wait();
  q.copy(d_output, &output, 1, std::vector<sycl::event>());
  const auto end = std::chrono::steady_clock::now();
  if(i!=0)
    overall += (end - start);

  average_elapsed_ns +=
       ev.get_profiling_info<sycl::info::event_profiling::command_end>() - ev.get_profiling_info<sycl::info::event_profiling::command_start>();
  }
  q.wait();

  std::cout << " array size: " << size
            << " output: " << output / iter
            << " average kernel_time (ns): " << average_elapsed_ns / iter
            << " average application time (ns): "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(overall).count() / (double)(iter - 1)
            << std::endl;
return 0;
}
