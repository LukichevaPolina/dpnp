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
 * Example BS.
 *
 * This example shows simple usage of the DPNP C++ Backend library
 * to calculate black scholes algorithm like in Python version
 *
 * Possible compile line:
 * . /opt/intel/oneapi/setvars.sh
 * g++ -g dpnp/backend/examples/example_bs_async.cpp -Idpnp -Idpnp/backend/include -Ldpnp -Wl,-rpath='$ORIGIN'/dpnp -ldpnp_backend_c -o example_bs_async
 */

#include <cmath>
#include <iostream>

#include "dpnp_async.hpp"
#include "dpnp_iface.hpp"

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   const double rate,
                   const double vol,
                   double* call,
                   double* put,
                   const size_t size)
{
    const size_t ndim = 1;
    const size_t scalar_size = 1;

    double* mr = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    mr[0] = -rate;

    double* vol_vol_two = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    vol_vol_two[0] = vol * vol * 2;

    double* quarter = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    quarter[0] = 0.25;

    double* one = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    one[0] = 1.;

    double* half = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    half[0] = 0.5;

    double* P = price;
    double* S = strike;
    double* T = t;

    // p_div_s = P / S
    double* p_div_s = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* p_div_s_deps_out =
        dpnp_divide_c<double, double, double>(p_div_s, P, size, &size, ndim, S, size, &size, ndim, NULL);

    // a = np.log(p_div_s)
    double* a = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* a_deps_out = dpnp_log_c<double, double>(p_div_s, a, size, p_div_s_deps_out);

    // b = T * mr
    double* b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* b_deps_out =
        dpnp_multiply_c<double, double, double>(b, T, size, &size, ndim, mr, scalar_size, &scalar_size, ndim, NULL);

    // z = T * vol_vol_twos
    double* z = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* z_deps_out = dpnp_multiply_c<double, double, double>(
        z, T, size, &size, ndim, vol_vol_two, scalar_size, &scalar_size, ndim, NULL);

    // c = quarters * z
    double* c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* c_deps_out = dpnp_multiply_c<double, double, double>(
        c, quarter, scalar_size, &scalar_size, ndim, z, size, &size, ndim, NULL, z_deps_out);

    // sqrt_z = np.sqrt(z)
    double* sqrt_z = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* sqrt_z_deps_out = dpnp_sqrt_c<double, double>(z, sqrt_z, size, z_deps_out);

    // y = ones / np.sqrt(z)
    double* y = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* y_deps_out = dpnp_divide_c<double, double, double>(
        y, one, scalar_size, &scalar_size, ndim, sqrt_z, size, &size, ndim, NULL, sqrt_z_deps_out);

    // a_sub_b = a - b
    double* a_sub_b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* a_sub_b_deps_in = new Deps();
    a_sub_b_deps_in->add(a_deps_out);
    a_sub_b_deps_in->add(b_deps_out);
    Deps* a_sub_b_deps_out =
        dpnp_subtract_c<double, double, double>(a_sub_b, a, size, &size, ndim, b, size, &size, ndim, NULL, a_sub_b_deps_in);

    // a_sub_b_add_c = a_sub_b + c
    double* a_sub_b_add_c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* a_sub_b_add_c_deps_in = new Deps();
    a_sub_b_add_c_deps_in->add(a_sub_b_deps_out);
    a_sub_b_add_c_deps_in->add(c_deps_out);
    Deps* a_sub_b_add_c_deps_out = dpnp_add_c<double, double, double>(
        a_sub_b_add_c, a_sub_b, size, &size, ndim, c, size, &size, ndim, NULL, a_sub_b_add_c_deps_in);

    // w1 = a_sub_b_add_c * y
    double* w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* w1_deps_in = new Deps();
    w1_deps_in->add(a_sub_b_add_c_deps_out);
    w1_deps_in->add(y_deps_out);
    Deps* w1_deps_out = dpnp_multiply_c<double, double, double>(
        w1, a_sub_b_add_c, size, &size, ndim, y, size, &size, ndim, NULL, w1_deps_in);

    // a_sub_b_sub_c = a_sub_b - c
    double* a_sub_b_sub_c = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* a_sub_b_sub_c_deps_in = new Deps();
    a_sub_b_sub_c_deps_in->add(a_sub_b_deps_out);
    a_sub_b_sub_c_deps_in->add(c_deps_out);
    Deps* a_sub_b_sub_c_deps_out = dpnp_subtract_c<double, double, double>(
        a_sub_b_sub_c, a_sub_b, size, &size, ndim, c, size, &size, ndim, NULL, a_sub_b_sub_c_deps_in);

    // w2 = a_sub_b_sub_c * y
    double* w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* w2_deps_in = new Deps();
    w2_deps_in->add(a_sub_b_sub_c_deps_out);
    w2_deps_in->add(y_deps_out);
    Deps* w2_deps_out = dpnp_multiply_c<double, double, double>(
        w2, a_sub_b_sub_c, size, &size, ndim, y, size, &size, ndim, NULL, w2_deps_in);

    // erf_w1 = np.erf(w1)
    double* erf_w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* erf_w1_deps_out = dpnp_erf_c<double>(w1, erf_w1, size, w1_deps_out);

    // halfs_mul_erf_w1 = halfs * erf_w1
    double* halfs_mul_erf_w1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* halfs_mul_erf_w1_deps_out = dpnp_multiply_c<double, double, double>(
        halfs_mul_erf_w1, half, scalar_size, &scalar_size, ndim, erf_w1, size, &size, ndim, NULL, erf_w1_deps_out);

    // d1 = halfs + halfs_mul_erf_w1
    double* d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* d1_deps_out = dpnp_add_c<double, double, double>(d1,
                                                           half,
                                                           scalar_size,
                                                           &scalar_size,
                                                           ndim,
                                                           halfs_mul_erf_w1,
                                                           size,
                                                           &size,
                                                           ndim,
                                                           NULL,
                                                           halfs_mul_erf_w1_deps_out);

    // erf_w2 = np.erf(w2)
    double* erf_w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* erf_w2_deps_out = dpnp_erf_c<double>(w2, erf_w2, size, w2_deps_out);

    // halfs_mul_erf_w2 = halfs * erf_w2
    double* halfs_mul_erf_w2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* halfs_mul_erf_w2_deps_out = dpnp_multiply_c<double, double, double>(
        halfs_mul_erf_w2, half, scalar_size, &scalar_size, ndim, erf_w2, size, &size, ndim, NULL, erf_w2_deps_out);

    // d2 = halfs + halfs_mul_erf_w2
    double* d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* d2_deps_out = dpnp_add_c<double, double, double>(d2,
                                                           half,
                                                           scalar_size,
                                                           &scalar_size,
                                                           ndim,
                                                           halfs_mul_erf_w2,
                                                           size,
                                                           &size,
                                                           ndim,
                                                           NULL,
                                                           halfs_mul_erf_w2_deps_out);

    // exp_b = np.exp(b)
    double* exp_b = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* exp_b_deps_out = dpnp_exp_c<double, double>(b, exp_b, size, b_deps_out);

    // Se = exp_b * S
    double* Se = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* Se_deps_out = dpnp_multiply_c<double, double, double>(
        Se, exp_b, size, &size, ndim, S, size, &size, ndim, NULL, exp_b_deps_out);

    // P_mul_d1 = P * d1
    double* P_mul_d1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* P_mul_d1_deps_out = dpnp_multiply_c<double, double, double>(
        P_mul_d1, P, size, &size, ndim, d1, size, &size, ndim, NULL, d1_deps_out);

    // Se_mul_d2 = Se * d2
    double* Se_mul_d2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* Se_mul_d2_deps_in = new Deps();
    Se_mul_d2_deps_in->add(Se_deps_out);
    Se_mul_d2_deps_in->add(d2_deps_out);
    Deps* Se_mul_d2_deps_out = dpnp_multiply_c<double, double, double>(
        Se_mul_d2, Se, size, &size, ndim, d2, size, &size, ndim, NULL, Se_mul_d2_deps_in);

    // r = P_mul_d1 - Se_mul_d2
    double* r = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* r_deps_in = new Deps();
    r_deps_in->add(P_mul_d1_deps_out);
    r_deps_in->add(Se_mul_d2_deps_out);
    Deps* r_deps_out = dpnp_subtract_c<double, double, double>(
        r, P_mul_d1, size, &size, ndim, Se_mul_d2, size, &size, ndim, NULL, r_deps_in);

    // call[:] = r
    Deps* copy_r_deps = dpnp_copyto_c<double, double>(call, r, size, r_deps_out);
    copy_r_deps->wait();

    // r_sub_P = r - P
    double* r_sub_P = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* r_sub_P_deps_out =
        dpnp_subtract_c<double, double, double>(r_sub_P, r, size, &size, ndim, P, size, &size, ndim, NULL, r_deps_out);

    // r_sub_P_add_Se = r_sub_P + Se
    double* r_sub_P_add_Se = (double*)dpnp_memory_alloc_c(size * sizeof(double));
    Deps* r_sub_P_add_Se_deps_in = new Deps();
    r_sub_P_add_Se_deps_in->add(r_sub_P_deps_out);
    r_sub_P_add_Se_deps_in->add(Se_deps_out);
    Deps* r_sub_P_add_Se_deps_out = dpnp_add_c<double, double, double>(
        r_sub_P_add_Se, r_sub_P, size, &size, ndim, Se, size, &size, ndim, NULL, r_sub_P_add_Se_deps_in);

    // put[:] = r_sub_P_add_Se
    Deps* copy_r_sub_P_add_Se_deps =
        dpnp_copyto_c<double, double>(put, r_sub_P_add_Se, size, r_sub_P_add_Se_deps_out);
    copy_r_sub_P_add_Se_deps->wait();

    dpnp_memory_free_c(p_div_s);
    dpnp_memory_free_c(mr);
    dpnp_memory_free_c(vol_vol_two);
    dpnp_memory_free_c(quarter);

    dpnp_memory_free_c(z);
    dpnp_memory_free_c(sqrt_z);
    dpnp_memory_free_c(one);
    dpnp_memory_free_c(a);
    dpnp_memory_free_c(a_sub_b_add_c);

    dpnp_memory_free_c(a_sub_b);
    dpnp_memory_free_c(c);
    dpnp_memory_free_c(a_sub_b_sub_c);

    dpnp_memory_free_c(y);
    dpnp_memory_free_c(w1);
    dpnp_memory_free_c(erf_w1);
    dpnp_memory_free_c(halfs_mul_erf_w1);
    dpnp_memory_free_c(w2);

    dpnp_memory_free_c(erf_w2);

    dpnp_memory_free_c(halfs_mul_erf_w2);
    dpnp_memory_free_c(half);
    dpnp_memory_free_c(exp_b);
    dpnp_memory_free_c(b);
    dpnp_memory_free_c(d1);

    dpnp_memory_free_c(d2);
    dpnp_memory_free_c(Se_mul_d2);
    dpnp_memory_free_c(P_mul_d1);
    dpnp_memory_free_c(r);
    dpnp_memory_free_c(r_sub_P);
    dpnp_memory_free_c(Se);

    dpnp_memory_free_c(r_sub_P_add_Se);
}

int main(int, char**)
{
    const size_t SIZE = 256;

    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;

    dpnp_queue_initialize_c(QueueOptions::GPU_SELECTOR);
    std::cout << "SYCL queue is CPU: " << dpnp_queue_is_cpu_c() << std::endl;

    double* price = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* strike = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* t = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_rng_srand_c(SEED);                           // np.random.seed(SEED)
    dpnp_rng_uniform_c<double>(price, PL, PH, SIZE);  // np.random.uniform(PL, PH, SIZE)
    dpnp_rng_uniform_c<double>(strike, SL, SH, SIZE); // np.random.uniform(SL, SH, SIZE)
    dpnp_rng_uniform_c<double>(t, TL, TH, SIZE);      // np.random.uniform(TL, TH, SIZE)

    std::cout << std::endl;
    double* zero = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    zero[0] = 0.;

    double* mone = (double*)dpnp_memory_alloc_c(1 * sizeof(double));
    mone[0] = -1.;

    double* call = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));
    double* put = (double*)dpnp_memory_alloc_c(SIZE * sizeof(double));

    dpnp_full_c<double>(zero, call, SIZE); // np.full(SIZE, 0., dtype=DTYPE)
    dpnp_full_c<double>(mone, put, SIZE);  // np.full(SIZE, -1., dtype=DTYPE)

    dpnp_memory_free_c(mone);
    dpnp_memory_free_c(zero);

    black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE);

    std::cout << "call: ";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << call[i] << ", ";
    }
    std::cout << "..." << std::endl;
    std::cout << "put: ";
    for (size_t i = 0; i < 10; ++i)
    {
        std::cout << put[i] << ", ";
    }
    std::cout << "..." << std::endl;

    dpnp_memory_free_c(put);
    dpnp_memory_free_c(call);

    dpnp_memory_free_c(t);
    dpnp_memory_free_c(strike);
    dpnp_memory_free_c(price);

    return 0;
}
