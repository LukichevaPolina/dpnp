//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <iostream>
#include "dpnp_async.hpp"
#include "dpnp_iface.hpp"

int main()
{
    const size_t size = 5;

    dpnp_queue_initialize_c();

    double* a = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    double* b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    double* c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    double* d = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    for (int i = 0; i < size; ++i)
    {
        a[i] = i;
        b[i] = 1.0;
        c[i] = 0;
        d[i] = 0;
        result[i] = 0;
    }

    Deps* deps_c_out = dpnp_add_c<double, double, double>(c, a, size, &size, 1, b, size, &size, 1, NULL);
    Deps* deps_d_out = dpnp_divide_c<double, double, double>(d, a, size, &size, 1, b, size, &size, 1, NULL);

    Deps* deps_result_in = new Deps();
    deps_result_in->add(deps_c_out);
    deps_result_in->add(deps_d_out);

    dpnp_multiply_c<double, double, double>(result, c, size, &size, 1, d, size, &size, 1, NULL, deps_result_in)->wait();

    for (int i = 0; i < size; ++i)
        std::cout << result[i] << " ";
}
