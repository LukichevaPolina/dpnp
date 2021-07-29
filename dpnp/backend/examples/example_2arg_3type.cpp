#include <iostream>
#include "dpnp_iface.hpp"
#include "dpnp_async.hpp"

int main()
{
  const size_t size = 5;

  dpnp_queue_initialize_c();

  double* a = (double*)dpnp_memory_alloc_c(size * sizeof(double));
  double* b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
  double* c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
  double* d = (double*)dpnp_memory_alloc_c(size * sizeof(double));
  double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

  for (int i = 0; i < size ; ++i)
  {
        a[i] = i;
        b[i] = 1.0;
        c[i] = 0;
        d[i] = 0;
        result[i] = 0;
  }

  Deps* deps_c_out = dpnp_add_c<double, double, double>(c, a, size, &size, 1, b, size, &size, 1, NULL);
  Deps* deps_d_out = dpnp_divide_c<double, double, double>(d, a, size, &size, 1, b, size, &size, 1, NULL, new Deps());

  Deps* deps_result_in = new Deps();
  deps_result_in->add(deps_c_out);
  deps_result_in->add(deps_d_out);
  
  dpnp_multiply_c<double, double, double>(result, c, size, &size, 1, d, size, &size, 1, NULL, deps_result_in)->wait();

  for (int i = 0; i < size; ++i)
    std::cout << result[i] << " ";

}