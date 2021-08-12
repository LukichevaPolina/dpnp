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

#include <cmath>
#include <iostream>

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

#include "dpnp_async.hpp"
#include "dpnp_async_pimpl.hpp"

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>                                \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    Deps* __name__(void* array1_in, void* result1, size_t size, Deps* deps_in)                                         \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        Deps* deps_out = new Deps();                                                                                   \
                                                                                                                       \
        _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);                                       \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);                                       \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType_output input_elem = array1[i];                                                               \
                result[i] = __operation1__;                                                                            \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.depends_on(deps_in->get_pImpl()->get());                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType_input, _DataType_output>>(gws,                          \
                                                                                         kernel_parallel_for_func);    \
        };                                                                                                             \
                                                                                                                       \
        if constexpr (std::is_same<_DataType_input, double>::value || std::is_same<_DataType_input, float>::value)     \
        {                                                                                                              \
            event = __operation2__;                                                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        deps_out->get_pImpl()->add(event);                                                                             \
        return deps_out;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    Deps* __name__(void* array1_in, void* result1, size_t size)                                                        \
    {                                                                                                                  \
        return __name__<_DataType_input, _DataType_output>(array1_in, result1, size, new Deps());                      \
    }

#include <dpnp_gen_1arg_2type_tbl.hpp>

template Deps* dpnp_acos_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_acos_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_acos_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_acos_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_acos_default_c)(void*, void*, size_t) = dpnp_acos_c<_DataType1, _DataType2>;

template Deps* dpnp_acosh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_acosh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_acosh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_acosh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_acosh_default_c)(void*, void*, size_t) = dpnp_acosh_c<_DataType1, _DataType2>;

template Deps* dpnp_asin_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_asin_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_asin_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_asin_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_asin_default_c)(void*, void*, size_t) = dpnp_asin_c<_DataType1, _DataType2>;

template Deps* dpnp_asinh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_asinh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_asinh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_asinh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_asinh_default_c)(void*, void*, size_t) = dpnp_asinh_c<_DataType1, _DataType2>;

template Deps* dpnp_atan_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_atan_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_atan_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_atan_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_atan_default_c)(void*, void*, size_t) = dpnp_atan_c<_DataType1, _DataType2>;

template Deps* dpnp_atanh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_atanh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_atanh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_atanh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_atanh_default_c)(void*, void*, size_t) = dpnp_atanh_c<_DataType1, _DataType2>;

template Deps* dpnp_cbrt_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cbrt_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cbrt_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_cbrt_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_cbrt_default_c)(void*, void*, size_t) = dpnp_cbrt_c<_DataType1, _DataType2>;

template Deps* dpnp_ceil_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_ceil_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_ceil_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_ceil_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_ceil_default_c)(void*, void*, size_t) = dpnp_ceil_c<_DataType1, _DataType2>;

template Deps* __dpnp_copyto_c<bool, bool>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<bool, int>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<bool, long>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<bool, float>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<bool, double>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<int, bool>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<int, int>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<int, long>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<int, float>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<int, double>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<long, bool>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<long, int>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<long, long>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<long, float>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<long, double>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<float, bool>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<float, int>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<float, long>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<float, float>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<float, double>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<double, bool>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<double, int>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<double, long>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<double, float>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<double, double>(void*, void*, size_t, Deps*);
template Deps* __dpnp_copyto_c<std::complex<double>, std::complex<double>>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*__dpnp_copyto_default_c)(void*, void*, size_t) = __dpnp_copyto_c<_DataType1, _DataType2>;

template Deps* dpnp_cos_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cos_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cos_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_cos_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_cos_default_c)(void*, void*, size_t) = dpnp_cos_c<_DataType1, _DataType2>;

template Deps* dpnp_cosh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cosh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_cosh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_cosh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_cosh_default_c)(void*, void*, size_t) = dpnp_cosh_c<_DataType1, _DataType2>;

template Deps* dpnp_degrees_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_degrees_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_degrees_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_degrees_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_degrees_default_c)(void*, void*, size_t) = dpnp_degrees_c<_DataType1, _DataType2>;

template Deps* dpnp_ediff1d_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_ediff1d_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_ediff1d_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_ediff1d_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_ediff1d_default_c)(void*, void*, size_t) = dpnp_ediff1d_c<_DataType1, _DataType2>;

template Deps* dpnp_exp2_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp2_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp2_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp2_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_exp2_default_c)(void*, void*, size_t) = dpnp_exp2_c<_DataType1, _DataType2>;

template Deps* dpnp_exp_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_exp_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_exp_default_c)(void*, void*, size_t) = dpnp_exp_c<_DataType1, _DataType2>;

template Deps* dpnp_expm1_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_expm1_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_expm1_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_expm1_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_expm1_default_c)(void*, void*, size_t) = dpnp_expm1_c<_DataType1, _DataType2>;

template Deps* dpnp_fabs_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_fabs_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_fabs_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_fabs_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_fabs_default_c)(void*, void*, size_t) = dpnp_fabs_c<_DataType1, _DataType2>;

template Deps* dpnp_floor_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_floor_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_floor_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_floor_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_floor_default_c)(void*, void*, size_t) = dpnp_floor_c<_DataType1, _DataType2>;

template Deps* dpnp_log10_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log10_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log10_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_log10_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_log10_default_c)(void*, void*, size_t) = dpnp_log10_c<_DataType1, _DataType2>;

template Deps* dpnp_log1p_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log1p_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log1p_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_log1p_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_log1p_default_c)(void*, void*, size_t) = dpnp_log1p_c<_DataType1, _DataType2>;

template Deps* dpnp_log2_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log2_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log2_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_log2_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_log2_default_c)(void*, void*, size_t) = dpnp_log2_c<_DataType1, _DataType2>;

template Deps* dpnp_log_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_log_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_log_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_log_default_c)(void*, void*, size_t) = dpnp_log_c<_DataType1, _DataType2>;

template Deps* dpnp_radians_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_radians_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_radians_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_radians_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_radians_default_c)(void*, void*, size_t) = dpnp_radians_c<_DataType1, _DataType2>;

template Deps* dpnp_sin_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sin_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sin_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_sin_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_sin_default_c)(void*, void*, size_t) = dpnp_sin_c<_DataType1, _DataType2>;

template Deps* dpnp_sinh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sinh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sinh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_sinh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_sinh_default_c)(void*, void*, size_t) = dpnp_sinh_c<_DataType1, _DataType2>;

template Deps* dpnp_sqrt_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sqrt_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_sqrt_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_sqrt_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_sqrt_default_c)(void*, void*, size_t) = dpnp_sqrt_c<_DataType1, _DataType2>;

template Deps* dpnp_tan_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_tan_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_tan_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_tan_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_tan_default_c)(void*, void*, size_t) = dpnp_tan_c<_DataType1, _DataType2>;

template Deps* dpnp_tanh_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_tanh_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_tanh_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_tanh_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_tanh_default_c)(void*, void*, size_t) = dpnp_tanh_c<_DataType1, _DataType2>;

template Deps* dpnp_trunc_c<int, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_trunc_c<long, double>(void*, void*, size_t, Deps*);
template Deps* dpnp_trunc_c<float, float>(void*, void*, size_t, Deps*);
template Deps* dpnp_trunc_c<double, double>(void*, void*, size_t, Deps*);

template <typename _DataType1, typename _DataType2>
Deps* (*dpnp_trunc_default_c)(void*, void*, size_t) = dpnp_trunc_c<_DataType1, _DataType2>;

static void func_map_init_elemwise_1arg_2type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acos_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acos_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acos_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acos_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acosh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acosh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acosh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acosh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asin_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asin_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asin_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asin_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asinh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asinh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asinh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asinh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atan_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atan_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atan_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atan_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atanh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atanh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atanh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atanh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cbrt_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cbrt_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cbrt_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cbrt_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_ceil_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_ceil_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ceil_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ceil_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_BLN] = {eft_BLN, (void*)__dpnp_copyto_default_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_INT] = {eft_INT, (void*)__dpnp_copyto_default_c<bool, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_LNG] = {eft_LNG, (void*)__dpnp_copyto_default_c<bool, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_FLT] = {eft_FLT, (void*)__dpnp_copyto_default_c<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_DBL] = {eft_DBL, (void*)__dpnp_copyto_default_c<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_BLN] = {eft_BLN, (void*)__dpnp_copyto_default_c<int, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_INT] = {eft_INT, (void*)__dpnp_copyto_default_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_LNG] = {eft_LNG, (void*)__dpnp_copyto_default_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_FLT] = {eft_FLT, (void*)__dpnp_copyto_default_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_DBL] = {eft_DBL, (void*)__dpnp_copyto_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_BLN] = {eft_BLN, (void*)__dpnp_copyto_default_c<long, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_INT] = {eft_INT, (void*)__dpnp_copyto_default_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_LNG] = {eft_LNG, (void*)__dpnp_copyto_default_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_FLT] = {eft_FLT, (void*)__dpnp_copyto_default_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_DBL] = {eft_DBL, (void*)__dpnp_copyto_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_BLN] = {eft_BLN, (void*)__dpnp_copyto_default_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_INT] = {eft_INT, (void*)__dpnp_copyto_default_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_LNG] = {eft_LNG, (void*)__dpnp_copyto_default_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_FLT] = {eft_FLT, (void*)__dpnp_copyto_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_DBL] = {eft_DBL, (void*)__dpnp_copyto_default_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_BLN] = {eft_BLN, (void*)__dpnp_copyto_default_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_INT] = {eft_INT, (void*)__dpnp_copyto_default_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_LNG] = {eft_LNG, (void*)__dpnp_copyto_default_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_FLT] = {eft_FLT, (void*)__dpnp_copyto_default_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_DBL] = {eft_DBL, (void*)__dpnp_copyto_default_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_C128][eft_C128] = {
        eft_C128, (void*)__dpnp_copyto_default_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cos_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cos_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cos_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cos_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cosh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cosh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cosh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cosh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_degrees_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_degrees_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_degrees_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_degrees_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_ediff1d_default_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ediff1d_default_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ediff1d_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ediff1d_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp2_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp2_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp2_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp2_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_expm1_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_expm1_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_expm1_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_expm1_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_fabs_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_fabs_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fabs_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fabs_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_floor_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_floor_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_floor_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_floor_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log10_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log10_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log10_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log10_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log1p_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log1p_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log1p_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log1p_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log2_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log2_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log2_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log2_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_radians_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_radians_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_radians_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_radians_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sin_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sin_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sin_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sin_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sinh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sinh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sinh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sinh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sqrt_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sqrt_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sqrt_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sqrt_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tan_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tan_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tan_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tan_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tanh_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tanh_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tanh_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tanh_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_trunc_default_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_trunc_default_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trunc_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trunc_default_c<double, double>};

    return;
}

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)                                                  \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    Deps* __name__(void* array1_in, void* result1, size_t size, Deps* deps_in)                                         \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        Deps* deps_out = new Deps();                                                                                   \
                                                                                                                       \
        if (!size)                                                                                                     \
        {                                                                                                              \
            return new Deps();                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);                                                   \
        _DataType* result = reinterpret_cast<_DataType*>(result1);                                                     \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType input_elem = array1[i];                                                                      \
                result[i] = __operation1__;                                                                            \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.depends_on(deps_in->get_pImpl()->get());                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                       \
        };                                                                                                             \
                                                                                                                       \
        if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)                 \
        {                                                                                                              \
            event = __operation2__;                                                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        deps_out->get_pImpl()->add(event);                                                                             \
        return deps_out;                                                                                               \
    }                                                                                                                  \
    template <typename _DataType>                                                                                      \
    Deps* __name__(void* array1_in, void* result1, size_t size)                                                        \
    {                                                                                                                  \
        return __name__<_DataType>(array1_in, result1, size, new Deps());                                              \
    }

#include <dpnp_gen_1arg_1type_tbl.hpp>

template Deps* dpnp_conjugate_c<std::complex<double>>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_conjugate_default_c)(void*, void*, size_t) = dpnp_conjugate_c<_DataType>;

template Deps* dpnp_copy_c<bool>(void*, void*, size_t, Deps*);
template Deps* dpnp_copy_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_copy_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_copy_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_copy_c<double>(void*, void*, size_t, Deps*);
template Deps* dpnp_copy_c<std::complex<double>>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_copy_default_c)(void*, void*, size_t) = dpnp_copy_c<_DataType>;

template Deps* dpnp_erf_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_erf_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_erf_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_erf_c<double>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_erf_default_c)(void*, void*, size_t) = dpnp_erf_c<_DataType>;

template Deps* dpnp_negative_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_negative_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_negative_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_negative_c<double>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_negative_default_c)(void*, void*, size_t) = dpnp_negative_c<_DataType>;

template Deps* dpnp_recip_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_recip_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_recip_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_recip_c<double>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_recip_default_c)(void*, void*, size_t) = dpnp_recip_c<_DataType>;

template Deps* dpnp_sign_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_sign_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_sign_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_sign_c<double>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_sign_default_c)(void*, void*, size_t) = dpnp_sign_c<_DataType>;

template Deps* dpnp_square_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_square_c<long>(void*, void*, size_t, Deps*);
template Deps* dpnp_square_c<float>(void*, void*, size_t, Deps*);
template Deps* dpnp_square_c<double>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* (*dpnp_square_default_c)(void*, void*, size_t) = dpnp_square_c<_DataType>;

static void func_map_init_elemwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_conjugate_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copy_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_copy_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ERF][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_erf_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_erf_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_erf_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_erf_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copy_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_default_c<double>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_C128][eft_C128] = {eft_C128,
                                                               (void*)dpnp_copy_default_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_negative_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_negative_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_negative_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_negative_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_recip_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_recip_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_recip_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_recip_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_sign_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sign_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sign_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sign_default_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_square_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_square_default_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_square_default_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_square_default_c<double>};

    return;
}

#define  MACRO_2ARG_3TYPES_OP(__name__, __operation1__, __operation2__)                                                \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class __name__##_kernel;                                                                                           \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class __name__##_opt_kernel;                                                                                       \
                                                                                                                       \
    template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>                         \
    Deps* __name__(void* result_out,                                                                                   \
                   const void* input1_in,                                                                              \
                   const size_t input1_size,                                                                           \
                   const size_t* input1_shape,                                                                         \
                   const size_t input1_shape_ndim,                                                                     \
                   const void* input2_in,                                                                              \
                   const size_t input2_size,                                                                           \
                   const size_t* input2_shape,                                                                         \
                   const size_t input2_shape_ndim,                                                                     \
                   const size_t* where,                                                                                \
                   Deps* deps_in)                                                                                      \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size || !input2_size)                                                                              \
        {                                                                                                              \
            return new Deps();                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        _DataType_input1* input1_data = reinterpret_cast<_DataType_input1*>(const_cast<void*>(input1_in));             \
        _DataType_input2* input2_data = reinterpret_cast<_DataType_input2*>(const_cast<void*>(input2_in));             \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);                                    \
                                                                                                                       \
        std::vector<size_t> result_shape =                                                                             \
            get_result_shape(input1_shape, input1_shape_ndim, input2_shape, input2_shape_ndim);                        \
                                                                                                                       \
        DPNPC_id<_DataType_input1>* input1_it;                                                                         \
        const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);                                     \
        input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(input1_it_size_in_bytes));       \
        new (input1_it) DPNPC_id<_DataType_input1>(input1_data, input1_shape, input1_shape_ndim);                      \
                                                                                                                       \
        input1_it->broadcast_to_shape(result_shape);                                                                   \
                                                                                                                       \
        DPNPC_id<_DataType_input2>* input2_it;                                                                         \
        const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);                                     \
        input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(input2_it_size_in_bytes));       \
        new (input2_it) DPNPC_id<_DataType_input2>(input2_data, input2_shape, input2_shape_ndim);                      \
                                                                                                                       \
        input2_it->broadcast_to_shape(result_shape);                                                                   \
        const size_t result_size = input1_it->get_output_size();                                                       \
                                                                                                                       \
        cl::sycl::range<1> gws(result_size);                                                                           \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            const size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                                \
            const _DataType_output input1_elem = (*input1_it)[i];                                                      \
            const _DataType_output input2_elem = (*input2_it)[i];                                                      \
            result[i] = __operation1__;                                                                                \
        };                                                                                                             \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.depends_on(deps_in->get_pImpl()->get());                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(           \
                gws, kernel_parallel_for_func);                                                                        \
        };                                                                                                             \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        Deps* deps_out = new Deps();                                                                                   \
                                                                                                                       \
        if (input1_size == input2_size)                                                                                \
        {                                                                                                              \
            if constexpr ((std::is_same<_DataType_input1, double>::value ||                                            \
                           std::is_same<_DataType_input1, float>::value) &&                                            \
                          std::is_same<_DataType_input2, _DataType_input1>::value)                                     \
            {                                                                                                          \
                event = __operation2__(                                                                                \
                    DPNP_QUEUE, result_size, input1_data, input2_data, result, deps_in->get_pImpl()->get());           \
                deps_out->get_pImpl()->add(event);                                                                     \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                event = DPNP_QUEUE.submit(kernel_func);                                                                \
                event.wait();                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        else if (input1_size == 1 || input2_size == 1)                                                                 \
        {                                                                                                              \
            cl::sycl::range<1> opt_gws(result_size);                                                                   \
            auto opt_kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                       \
                const size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                            \
                _DataType_output input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[i];                   \
                _DataType_output input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[i];                   \
                result[i] = __operation1__;                                                                            \
            };                                                                                                         \
            auto opt_kernel_func = [&](cl::sycl::handler& cgh) {                                                       \
                cgh.depends_on(deps_in->get_pImpl()->get());                                                           \
                cgh.parallel_for<class __name__##_opt_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(   \
                    opt_gws, opt_kernel_parallel_for_func);                                                            \
            };                                                                                                         \
            event = DPNP_QUEUE.submit(opt_kernel_func);                                                                \
            deps_out->get_pImpl()->add(event);                                                                         \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        input1_it->~DPNPC_id();                                                                                        \
        input2_it->~DPNPC_id();                                                                                        \
        return deps_out;                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>                         \
    Deps* __name__(void* result_out,                                                                                   \
                   const void* input1_in,                                                                              \
                   const size_t input1_size,                                                                           \
                   const size_t* input1_shape,                                                                         \
                   const size_t input1_shape_ndim,                                                                     \
                   const void* input2_in,                                                                              \
                   const size_t input2_size,                                                                           \
                   const size_t* input2_shape,                                                                         \
                   const size_t input2_shape_ndim,                                                                     \
                   const size_t* where)                                                                                \
    {                                                                                                                  \
        return __name__<_DataType_output, _DataType_input1, _DataType_input2>(result_out,                              \
                                                                              input1_in,                               \
                                                                              input1_size,                             \
                                                                              input1_shape,                            \
                                                                              input1_shape_ndim,                       \
                                                                              input2_in,                               \
                                                                              input2_size,                             \
                                                                              input2_shape,                            \
                                                                              input2_shape_ndim,                       \
                                                                              where,                                   \
                                                                              new Deps());                             \
    }

#include <dpnp_gen_2arg_3type_tbl.hpp>

template Deps* dpnp_add_c<int, int, int>(void*,
                                         const void*,
                                         const size_t,
                                         const size_t*,
                                         const size_t,
                                         const void*,
                                         const size_t,
                                         const size_t*,
                                         const size_t,
                                         const size_t*,
                                         Deps*);
template Deps* dpnp_add_c<long, int, long>(void*,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const size_t*,
                                           Deps*);
template Deps* dpnp_add_c<double, int, float>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_add_c<double, int, double>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_add_c<long, long, int>(void*,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const size_t*,
                                           Deps*);
template Deps* dpnp_add_c<long, long, long>(void*,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const size_t*,
                                            Deps*);
template Deps* dpnp_add_c<double, long, float>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_add_c<double, long, double>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_add_c<double, float, int>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_add_c<double, float, long>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_add_c<float, float, float>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_add_c<double, float, double>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_add_c<double, double, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_add_c<double, double, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_add_c<double, double, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_add_c<double, double, double>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_add_default_c)(void*,
                            const void*,
                            const size_t,
                            const size_t*,
                            const size_t,
                            const void*,
                            const size_t,
                            const size_t*,
                            const size_t,
                            const size_t*) = dpnp_add_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_arctan2_c<double, int, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_arctan2_c<double, int, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_arctan2_c<double, int, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_arctan2_c<double, int, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_arctan2_c<double, long, int>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_arctan2_c<double, long, long>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_arctan2_c<double, long, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_arctan2_c<double, long, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_arctan2_c<double, float, int>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_arctan2_c<double, float, long>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_arctan2_c<float, float, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_arctan2_c<double, float, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_arctan2_c<double, double, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_arctan2_c<double, double, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_arctan2_c<double, double, float>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_arctan2_c<double, double, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_arctan2_default_c)(void*,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const size_t*) = dpnp_arctan2_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_copysign_c<double, int, int>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_copysign_c<double, int, long>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_copysign_c<double, int, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_copysign_c<double, int, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_copysign_c<double, long, int>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_copysign_c<double, long, long>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_copysign_c<double, long, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_copysign_c<double, long, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_copysign_c<double, float, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_copysign_c<double, float, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_copysign_c<float, float, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_copysign_c<double, float, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_copysign_c<double, double, int>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_copysign_c<double, double, long>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_copysign_c<double, double, float>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_copysign_c<double, double, double>(void*,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const size_t*,
                                                       Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_copysign_default_c)(void*,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const size_t*) = dpnp_copysign_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_divide_c<double, int, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_divide_c<double, int, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_divide_c<double, int, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_divide_c<double, int, double>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_divide_c<double, long, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_divide_c<double, long, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_divide_c<double, long, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_divide_c<double, long, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_divide_c<double, float, int>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_divide_c<double, float, long>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_divide_c<float, float, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_divide_c<double, float, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_divide_c<double, double, int>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_divide_c<double, double, long>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_divide_c<double, double, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_divide_c<double, double, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_divide_default_c)(void*,
                               const void*,
                               const size_t,
                               const size_t*,
                               const size_t,
                               const void*,
                               const size_t,
                               const size_t*,
                               const size_t,
                               const size_t*) = dpnp_divide_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_fmod_c<int, int, int>(void*,
                                          const void*,
                                          const size_t,
                                          const size_t*,
                                          const size_t,
                                          const void*,
                                          const size_t,
                                          const size_t*,
                                          const size_t,
                                          const size_t*,
                                          Deps*);
template Deps* dpnp_fmod_c<long, int, long>(void*,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const size_t*,
                                            Deps*);
template Deps* dpnp_fmod_c<double, int, float>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_fmod_c<double, int, double>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_fmod_c<long, long, int>(void*,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const void*,
                                            const size_t,
                                            const size_t*,
                                            const size_t,
                                            const size_t*,
                                            Deps*);
template Deps* dpnp_fmod_c<long, long, long>(void*,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const size_t*,
                                             Deps*);
template Deps* dpnp_fmod_c<double, long, float>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_fmod_c<double, long, double>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_fmod_c<double, float, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_fmod_c<double, float, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_fmod_c<float, float, float>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_fmod_c<double, float, double>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_fmod_c<double, double, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_fmod_c<double, double, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_fmod_c<double, double, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_fmod_c<double, double, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_fmod_default_c)(void*,
                             const void*,
                             const size_t,
                             const size_t*,
                             const size_t,
                             const void*,
                             const size_t,
                             const size_t*,
                             const size_t,
                             const size_t*) = dpnp_fmod_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_hypot_c<double, int, int>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_hypot_c<double, int, long>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_hypot_c<double, int, float>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_hypot_c<double, int, double>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_hypot_c<double, long, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_hypot_c<double, long, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_hypot_c<double, long, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_hypot_c<double, long, double>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_hypot_c<double, float, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_hypot_c<double, float, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_hypot_c<float, float, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_hypot_c<double, float, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_hypot_c<double, double, int>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_hypot_c<double, double, long>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_hypot_c<double, double, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_hypot_c<double, double, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_hypot_default_c)(void*,
                              const void*,
                              const size_t,
                              const size_t*,
                              const size_t,
                              const void*,
                              const size_t,
                              const size_t*,
                              const size_t,
                              const size_t*) = dpnp_hypot_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_maximum_c<int, int, int>(void*,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const size_t*,
                                             Deps*);
template Deps* dpnp_maximum_c<long, int, long>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_maximum_c<double, int, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_maximum_c<double, int, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_maximum_c<long, long, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_maximum_c<long, long, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_maximum_c<double, long, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_maximum_c<double, long, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_maximum_c<double, float, int>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_maximum_c<double, float, long>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_maximum_c<float, float, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_maximum_c<double, float, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_maximum_c<double, double, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_maximum_c<double, double, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_maximum_c<double, double, float>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_maximum_c<double, double, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_maximum_default_c)(void*,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const size_t*) = dpnp_maximum_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_minimum_c<int, int, int>(void*,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const size_t*,
                                             Deps*);
template Deps* dpnp_minimum_c<long, int, long>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_minimum_c<double, int, float>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_minimum_c<double, int, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_minimum_c<long, long, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_minimum_c<long, long, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_minimum_c<double, long, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_minimum_c<double, long, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_minimum_c<double, float, int>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_minimum_c<double, float, long>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_minimum_c<float, float, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_minimum_c<double, float, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_minimum_c<double, double, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_minimum_c<double, double, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_minimum_c<double, double, float>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_minimum_c<double, double, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_minimum_default_c)(void*,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const void*,
                                const size_t,
                                const size_t*,
                                const size_t,
                                const size_t*) = dpnp_minimum_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_multiply_c<bool, bool, bool>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_multiply_c<int, bool, int>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_multiply_c<long, bool, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_multiply_c<float, bool, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_multiply_c<double, bool, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_multiply_c<int, int, bool>(void*,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const void*,
                                               const size_t,
                                               const size_t*,
                                               const size_t,
                                               const size_t*,
                                               Deps*);
template Deps* dpnp_multiply_c<int, int, int>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_multiply_c<long, int, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_multiply_c<double, int, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_multiply_c<double, int, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_multiply_c<long, long, bool>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_multiply_c<long, long, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_multiply_c<long, long, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_multiply_c<double, long, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_multiply_c<float, float, bool>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_multiply_c<double, float, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_multiply_c<double, float, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_multiply_c<float, float, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_multiply_c<double, float, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_multiply_c<double, double, bool>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_multiply_c<double, double, int>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_multiply_c<double, double, long>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_multiply_c<double, double, float>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_multiply_c<double, double, double>(void*,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const size_t*,
                                                       Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_multiply_default_c)(void*,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const size_t*) = dpnp_multiply_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_power_c<int, int, int>(void*,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const void*,
                                           const size_t,
                                           const size_t*,
                                           const size_t,
                                           const size_t*,
                                           Deps*);
template Deps* dpnp_power_c<long, int, long>(void*,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const size_t*,
                                             Deps*);
template Deps* dpnp_power_c<double, int, float>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_power_c<double, int, double>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_power_c<long, long, int>(void*,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const void*,
                                             const size_t,
                                             const size_t*,
                                             const size_t,
                                             const size_t*,
                                             Deps*);
template Deps* dpnp_power_c<long, long, long>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_power_c<double, long, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_power_c<double, long, double>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_power_c<double, float, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_power_c<double, float, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_power_c<float, float, float>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_power_c<double, float, double>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_power_c<double, double, int>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_power_c<double, double, long>(void*,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const void*,
                                                  const size_t,
                                                  const size_t*,
                                                  const size_t,
                                                  const size_t*,
                                                  Deps*);
template Deps* dpnp_power_c<double, double, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_power_c<double, double, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_power_default_c)(void*,
                              const void*,
                              const size_t,
                              const size_t*,
                              const size_t,
                              const void*,
                              const size_t,
                              const size_t*,
                              const size_t,
                              const size_t*) = dpnp_power_c<_DataType1, _DataType2, _DataType3>;

template Deps* dpnp_subtract_c<int, int, int>(void*,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const void*,
                                              const size_t,
                                              const size_t*,
                                              const size_t,
                                              const size_t*,
                                              Deps*);
template Deps* dpnp_subtract_c<long, int, long>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_subtract_c<double, int, float>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_subtract_c<double, int, double>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_subtract_c<long, long, int>(void*,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const void*,
                                                const size_t,
                                                const size_t*,
                                                const size_t,
                                                const size_t*,
                                                Deps*);
template Deps* dpnp_subtract_c<long, long, long>(void*,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const void*,
                                                 const size_t,
                                                 const size_t*,
                                                 const size_t,
                                                 const size_t*,
                                                 Deps*);
template Deps* dpnp_subtract_c<double, long, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_subtract_c<double, long, double>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_subtract_c<double, float, int>(void*,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const void*,
                                                   const size_t,
                                                   const size_t*,
                                                   const size_t,
                                                   const size_t*,
                                                   Deps*);
template Deps* dpnp_subtract_c<double, float, long>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_subtract_c<float, float, float>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_subtract_c<double, float, double>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_subtract_c<double, double, int>(void*,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const void*,
                                                    const size_t,
                                                    const size_t*,
                                                    const size_t,
                                                    const size_t*,
                                                    Deps*);
template Deps* dpnp_subtract_c<double, double, long>(void*,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const void*,
                                                     const size_t,
                                                     const size_t*,
                                                     const size_t,
                                                     const size_t*,
                                                     Deps*);
template Deps* dpnp_subtract_c<double, double, float>(void*,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const void*,
                                                      const size_t,
                                                      const size_t*,
                                                      const size_t,
                                                      const size_t*,
                                                      Deps*);
template Deps* dpnp_subtract_c<double, double, double>(void*,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const void*,
                                                       const size_t,
                                                       const size_t*,
                                                       const size_t,
                                                       const size_t*,
                                                       Deps*);

template <typename _DataType1, typename _DataType2, typename _DataType3>
Deps* (*dpnp_subtract_default_c)(void*,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const void*,
                                 const size_t,
                                 const size_t*,
                                 const size_t,
                                 const size_t*) = dpnp_subtract_c<_DataType1, _DataType2, _DataType3>;

static void func_map_init_elemwise_2arg_3type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_add_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_add_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_add_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_add_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_add_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_add_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_add_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_add_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_add_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_add_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_add_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_add_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_add_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_add_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_add_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_add_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_default_c<double, int, int>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_default_c<double, int, long>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_default_c<double, long, int>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)dpnp_arctan2_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_arctan2_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, int, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, int, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, long, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, long, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_FLT] = {eft_FLT,
                                                              (void*)dpnp_copysign_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_copysign_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_divide_default_c<double, int, int>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_default_c<double, int, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_divide_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_divide_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_divide_default_c<double, long, int>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_divide_default_c<double, long, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_divide_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL,
                                                            (void*)dpnp_divide_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_divide_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_divide_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL,
                                                            (void*)dpnp_divide_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_divide_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL,
                                                            (void*)dpnp_divide_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL,
                                                            (void*)dpnp_divide_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL,
                                                            (void*)dpnp_divide_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_fmod_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_fmod_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_fmod_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fmod_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_fmod_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, int, int>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_default_c<double, int, long>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, long, int>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_default_c<double, long, long>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_hypot_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_DBL] = {eft_DBL,
                                                           (void*)dpnp_hypot_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_maximum_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_maximum_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)dpnp_maximum_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_maximum_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_minimum_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_minimum_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_FLT] = {eft_FLT,
                                                             (void*)dpnp_minimum_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_INT] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_LNG] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_FLT] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_DBL] = {eft_DBL,
                                                             (void*)dpnp_minimum_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_BLN] = {eft_BLN,
                                                              (void*)dpnp_multiply_default_c<bool, bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_INT] = {eft_INT, (void*)dpnp_multiply_default_c<int, bool, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_LNG] = {eft_LNG,
                                                              (void*)dpnp_multiply_default_c<long, bool, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_FLT] = {eft_FLT,
                                                              (void*)dpnp_multiply_default_c<float, bool, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, bool, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_BLN] = {eft_INT, (void*)dpnp_multiply_default_c<int, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_multiply_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_BLN] = {eft_LNG,
                                                              (void*)dpnp_multiply_default_c<long, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_multiply_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {eft_LNG,
                                                              (void*)dpnp_multiply_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_BLN] = {eft_FLT,
                                                              (void*)dpnp_multiply_default_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {eft_FLT,
                                                              (void*)dpnp_multiply_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_BLN] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_multiply_default_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_multiply_default_c<std::complex<double>, std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_power_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_power_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_power_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_power_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_power_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_power_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_power_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_power_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_power_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_power_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_power_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_power_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_power_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_power_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_power_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_DBL] = {eft_DBL,
                                                           (void*)dpnp_power_default_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_subtract_default_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_subtract_default_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_subtract_default_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_LNG] = {eft_LNG,
                                                              (void*)dpnp_subtract_default_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_FLT] = {eft_FLT,
                                                              (void*)dpnp_subtract_default_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_INT] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_LNG] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_FLT] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_DBL] = {eft_DBL,
                                                              (void*)dpnp_subtract_default_c<double, double, double>};

    return;
}

void func_map_init_elemwise(func_map_t& fmap)
{
    func_map_init_elemwise_1arg_1type(fmap);
    func_map_init_elemwise_1arg_2type(fmap);
    func_map_init_elemwise_2arg_3type(fmap);

    return;
}
