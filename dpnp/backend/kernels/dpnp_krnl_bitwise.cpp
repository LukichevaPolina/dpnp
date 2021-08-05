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
#include "dpnp_async_pimpl.hpp"
#include "dpnp_fptr.hpp"
#include "dpnp_iface.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_invert_c_kernel;

template <typename _DataType>
Deps* dpnp_invert_c(void* array1_in, void* result1, size_t size, Deps* deps_in)
{
    cl::sycl::event event;
    Deps* deps_out = new Deps();
    _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
        {
            _DataType input_elem1 = array1[i];
            result[i] = ~input_elem1;
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.depends_on(deps_in->get_pImpl()->get());
        cgh.parallel_for<class dpnp_invert_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);
    deps_out->get_pImpl()->add(event);
    return deps_out;
}

template Deps* dpnp_invert_c<int>(void*, void*, size_t, Deps*);
template Deps* dpnp_invert_c<long>(void*, void*, size_t, Deps*);

template <typename _DataType>
Deps* dpnp_invert_c(void* array1_in, void* result1, size_t size)
{
    return dpnp_invert_c<_DataType>(array1_in, result1, size, new Deps());
}

template <typename _DataType>
Deps* (*dpnp_invert_default_c)(void*, void*, size_t) = dpnp_invert_c<_DataType>;

static void func_map_init_bitwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_invert_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_invert_default_c<long>};

    return;
}

#define MACRO_2ARG_1TYPE_ASYNC_OP(__name__, __operation__)                                                             \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
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
        (void)input1_shape;                                                                                            \
        (void)input1_shape_ndim;                                                                                       \
        (void)input2_shape;                                                                                            \
        (void)input2_shape_ndim;                                                                                       \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size || !input2_size)                                                                              \
        {                                                                                                              \
            return new Deps();                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        Deps* deps_out = new Deps();                                                                                   \
        const _DataType* input1 = reinterpret_cast<const _DataType*>(input1_in);                                       \
        const _DataType* input2 = reinterpret_cast<const _DataType*>(input2_in);                                       \
        _DataType* result = reinterpret_cast<_DataType*>(result_out);                                                  \
                                                                                                                       \
        const size_t gws_size = std::max(input1_size, input2_size);                                                    \
        cl::sycl::range<1> gws(gws_size);                                                                              \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            const _DataType input_elem1 = (input1_size == 1) ? input1[0] : input1[i];                                  \
            const _DataType input_elem2 = (input2_size == 1) ? input2[0] : input2[i];                                  \
            result[i] = __operation__;                                                                                 \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.depends_on(deps_in->get_pImpl()->get());                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                       \
        };                                                                                                             \
                                                                                                                       \
        event = DPNP_QUEUE.submit(kernel_func);                                                                        \
                                                                                                                       \
        deps_out->get_pImpl()->add(event);                                                                             \
        return deps_out;                                                                                               \
    }

#define MACRO_2ARG_1TYPE_OP(__name__, __operation__)                                                                   \
    template <typename _DataType>                                                                                      \
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
        return __name__<_DataType>(result_out,                                                                         \
                                   input1_in,                                                                          \
                                   input1_size,                                                                        \
                                   input1_shape,                                                                       \
                                   input1_shape_ndim,                                                                  \
                                   input2_in,                                                                          \
                                   input2_size,                                                                        \
                                   input2_shape,                                                                       \
                                   input2_shape_ndim,                                                                  \
                                   where,                                                                              \
                                   new Deps());                                                                        \
    }

#include <dpnp_gen_2arg_1type_async_tbl.hpp>
#include <dpnp_gen_2arg_1type_tbl.hpp>

template Deps* dpnp_bitwise_and_c<int>(void*,
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
template Deps* dpnp_bitwise_and_c<long>(void*,
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

template <typename _DataType>
Deps* (*dpnp_bitwise_and_default_c)(void*,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const size_t*) = dpnp_bitwise_and_c<_DataType>;

template Deps* dpnp_bitwise_or_c<int>(void*,
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

template Deps* dpnp_bitwise_or_c<long>(void*,
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

template <typename _DataType>
Deps* (*dpnp_bitwise_or_default_c)(void*,
                                   const void*,
                                   const size_t,
                                   const size_t*,
                                   const size_t,
                                   const void*,
                                   const size_t,
                                   const size_t*,
                                   const size_t,
                                   const size_t*) = dpnp_bitwise_or_c<_DataType>;

template Deps* dpnp_bitwise_xor_c<int>(void*,
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
template Deps* dpnp_bitwise_xor_c<long>(void*,
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

template <typename _DataType>
Deps* (*dpnp_bitwise_xor_default_c)(void*,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const size_t*) = dpnp_bitwise_xor_c<_DataType>;

template Deps* dpnp_left_shift_c<int>(void*,
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
template Deps* dpnp_left_shift_c<long>(void*,
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

template <typename _DataType>
Deps* (*dpnp_left_shift_default_c)(void*,
                                   const void*,
                                   const size_t,
                                   const size_t*,
                                   const size_t,
                                   const void*,
                                   const size_t,
                                   const size_t*,
                                   const size_t,
                                   const size_t*) = dpnp_left_shift_c<_DataType>;

template Deps* dpnp_right_shift_c<int>(void*,
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
template Deps* dpnp_right_shift_c<long>(void*,
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

template <typename _DataType>
Deps* (*dpnp_right_shift_default_c)(void*,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const void*,
                                    const size_t,
                                    const size_t*,
                                    const size_t,
                                    const size_t*) = dpnp_right_shift_c<_DataType>;

static void func_map_init_bitwise_2arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_and_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_and_default_c<long>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_or_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_or_default_c<long>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_xor_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_xor_default_c<long>};

    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_left_shift_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_left_shift_default_c<long>};

    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_right_shift_default_c<int>};
    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_right_shift_default_c<long>};

    return;
}

void func_map_init_bitwise(func_map_t& fmap)
{
    func_map_init_bitwise_1arg_1type(fmap);
    func_map_init_bitwise_2arg_1type(fmap);

    return;
}
