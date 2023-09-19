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
 *  @filename reduce_std_cpp
 *
 **************************************************************************/

#include <execution>
//#define SEQ std::execution::seq,
//#define PAR std::execution::par,
 
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

int main(int argc, char *argv[])
{
  using type_t = double;
  size_t iter = 2;
  if (argc > 1)
    iter = atoi(argv[1]);
  size_t size = 1 << 15;
  if (argc > 2)
    size = atoi(argv[2]);

  std::vector<type_t> input(size);
  std::iota(input.begin(), input.end(), 0);
  std::chrono::duration<double> overall{0};
  type_t sum {0};
  for (int i = 0; i < iter; i++)
  {
    const auto start = std::chrono::steady_clock::now();

    sum = std::reduce(std::execution::seq, input.begin(),
                           input.end(), 0, std::plus<>{});

    const auto end = std::chrono::steady_clock::now();
    overall += (end - start);
  }
  std::cout << " array size: " << size
            << " output: " << sum
            << " average application time (ns): "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(overall).count() / (double)(iter - 1)
            << std::endl;

  return 0;
  }
