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

/**
 * Example 10.
 *
 * Possible compile line:
 * dpcpp -fPIC examples/example10.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example10
 *
 */

#include <iostream>
#include <time.h>

#include <dpnp_iface.hpp>


int main(int, char**)
{
    const size_t size = 100000000;
    const size_t iters = 300;

    clock_t start, end;
    double dev_time_used, sum_dev_time_used = 0.0;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);

    double* result = (double*)dpnp_memory_alloc_c(size * sizeof(double));

    size_t seed = 10;
    double loc = 0.0;
    double scale = 1.0;

    dpnp_rng_srand_c(seed);

    std::cout << "Normal distr. params:\nloc is " << loc << ", scale is " << scale << std::endl;

    for (size_t i = 0; i < iters; ++i)
    {
        start = clock();
        dpnp_rng_normal_c<double>(result, loc, scale, size);
        end = clock();
        sum_dev_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
        // TODO: cumulative addition error
        // div error
    }

    dpnp_memory_free_c(result);

    dev_time_used = sum_dev_time_used / iters;
    std::cout << dev_time_used << std::endl;

    return 0;
}
