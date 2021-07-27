
#include <iostream>
#include "dpnp_iface.hpp"
#include "dpnp_async.hpp"

int main()
{
  const size_t size = 5;

  dpnp_queue_initialize_c();

  double* a = (double*)dpnp_memory_alloc_c(size * size * sizeof(double));
  double* b = (double*)dpnp_memory_alloc_c(size * size * sizeof(double));
  double* c = (double*)dpnp_memory_alloc_c(size * size * sizeof(double));
  double* d = (double*)dpnp_memory_alloc_c(size * size * sizeof(double));
  double* result = (double*)dpnp_memory_alloc_c(size * size * sizeof(double));

  for (int i = 0; i < size * size; ++i)
  {
        a[i] = 1.0 * i;
        b[i] = 1.0;
        c[i] = 0;
        d[i] = 0;
        result[i] = 0;
  }

  Deps* deps_c_out = dpnp_matmul_c<double>(a, b, c, size, size, size);
  Deps* deps_d_out = dpnp_matmul_c<double>(a, a, d, size, size, size);

  Deps* deps_result_in = new Deps();
  deps_result_in->add(deps_c_out);
  deps_result_in->add(deps_d_out);
  
  dpnp_matmul_c<double>(c, d, result, size, size, size, deps_result_in)->wait();

  for (int i = 0; i < size * size; ++i)
    std::cout << result[i] << " ";

}