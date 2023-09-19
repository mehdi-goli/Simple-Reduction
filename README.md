This is a very simple reduction algorithms testing various option in reducing a vector across different arcitecture using SYCL-2020 features.

To compile the code: 
* download [DPCPP] (https://github.com/intel/llvm) 

Compiling reduce_sycl_2020.cpp 
```bahs
 /path/to/dpcpp_compiler/bin/clang++ -fsycl reduce_sycl_2020.cpp -O3 -o reduce_sycl_2020
 ONEAPI_DEVICE_SELECTOR=*:gpu ./reduce_sycl_2020
```


Compiling reduce_oneDPL.cpp
 * downlode [oneDPL](https://github.com/oneapi-src/oneDPL)
```bash
/path/to/dpcpp_compiler/bin/clang++ -fsycl reduceonedpl.cpp -I /path/to/oneDPL/include -DONEDPL_USE_TBB_BACKEND=0 -DONEDPL_USE_DPCPP_BACKEND=1 -DPSTL_USE_PARALLEL_POLICIES=0 -O3 -o reduce_onedpl
ONEAPI_DEVICE_SELECTOR=*:gpu ./reduce_onedpl 
```

Compiling reduce_std_cpp.cpp
```bash
/path/to/clang++ -O3 reduce_std_cpp.cpp -o reduce_std_cpp
./reduce_std_cpp
```
