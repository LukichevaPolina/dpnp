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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

namespace mkl_blas = oneapi::mkl::blas::row_major;
namespace mkl_stats = oneapi::mkl::stats;

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_correlate_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_correlate_c(void* result_out,
                      const void* input1_in,
                      const size_t input1_size,
                      const shape_elem_type* input1_shape,
                      const size_t input1_shape_ndim,
                      const void* input2_in,
                      const size_t input2_size,
                      const shape_elem_type* input2_shape,
                      const size_t input2_shape_ndim,
                      const size_t* where)
{
    (void)where;

    shape_elem_type dummy[] = {1};
    dpnp_dot_c<_DataType_output, _DataType_input1, _DataType_input2>(result_out,
                                                                     42,   // dummy result_size
                                                                     42,   // dummy result_ndim
                                                                     NULL, // dummy result_shape
                                                                     NULL, // dummy result_strides
                                                                     input1_in,
                                                                     input1_size,
                                                                     input1_shape_ndim,
                                                                     input1_shape,
                                                                     dummy, // dummy input1_strides
                                                                     input2_in,
                                                                     input2_size,
                                                                     input2_shape_ndim,
                                                                     input2_shape,
                                                                     dummy); // dummy input2_strides

    return;
}

template <typename _DataType>
class dpnp_cov_c_kernel1;

template <typename _DataType>
class dpnp_cov_c_kernel2;

template <typename _DataType>
void dpnp_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols)
{
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, nrows * ncols);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if (!nrows || !ncols)
    {
        return;
    }

    auto policy = oneapi::dpl::execution::make_device_policy<class dpnp_cov_c_kernel1<_DataType>>(DPNP_QUEUE);

    _DataType* mean = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(nrows * sizeof(_DataType)));
    for (size_t i = 0; i < nrows; ++i)
    {
        _DataType* row_start = array_1 + ncols * i;
        mean[i] = std::reduce(policy, row_start, row_start + ncols, _DataType(0), std::plus<_DataType>()) / ncols;
    }
    policy.queue().wait();

    _DataType* temp = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(nrows * ncols * sizeof(_DataType)));
    for (size_t i = 0; i < nrows; ++i)
    {
        size_t offset = ncols * i;
        _DataType* row_start = array_1 + offset;
        std::transform(policy, row_start, row_start + ncols, temp + offset, [=](_DataType x) { return x - mean[i]; });
    }
    policy.queue().wait();

    sycl::event event_syrk;

    const _DataType alpha = _DataType(1) / (ncols - 1);
    const _DataType beta = _DataType(0);

    event_syrk = mkl_blas::syrk(DPNP_QUEUE,                       // queue &exec_queue,
                                oneapi::mkl::uplo::upper,         // uplo upper_lower,
                                oneapi::mkl::transpose::nontrans, // transpose trans,
                                nrows,                            // std::int64_t n,
                                ncols,                            // std::int64_t k,
                                alpha,                            // T alpha,
                                temp,                             //const T* a,
                                ncols,                            // std::int64_t lda,
                                beta,                             // T beta,
                                result,                           // T* c,
                                nrows);                           // std::int64_t ldc);
    event_syrk.wait();

    // fill lower elements
    sycl::event event;
    sycl::range<1> gws(nrows * nrows);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        const size_t row_idx = idx / nrows;
        const size_t col_idx = idx - row_idx * nrows;
        if (col_idx < row_idx)
        {
            result[idx] = result[col_idx * nrows + row_idx];
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_cov_c_kernel2<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    dpnp_memory_free_c(mean);
    dpnp_memory_free_c(temp);

    return;
}

template <typename _DataType_input, typename _DataType_output>
void dpnp_count_nonzero_c(void* array1_in, void* result1_out, size_t size)
{
    if (array1_in == nullptr)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, size, true);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result1_out, 1, true, true);
    _DataType_input* array1 = input1_ptr.get_ptr();
    _DataType_output* result1 = result_ptr.get_ptr();

    result1[0] = 0;

    for (size_t i = 0; i < size; ++i)
    {
        if (array1[i] != 0)
        {
            result1[0] += 1;
        }
    }

    return;
}

template <typename _DataType>
class dpnp_max_c_kernel;

template <typename _DataType>
void dpnp_max_c(void* array1_in,
                void* result1,
                const size_t result_size,
                const shape_elem_type* shape,
                size_t ndim,
                const shape_elem_type* axis,
                size_t naxis)
{
    const size_t size_input = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!size_input)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size_input, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, result_size, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    if (naxis == 0)
    {
        __attribute__((unused)) void* tmp = (void*)(axis + naxis);

        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i)
        {
            size *= shape[i];
        }

        if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
        {
            // Required initializing the result before call the function
            result[0] = array_1[0];

            auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size, array_1);

            sycl::event event = mkl_stats::max(DPNP_QUEUE, dataset, result);

            event.wait();
        }
        else
        {
            auto policy = oneapi::dpl::execution::make_device_policy<class dpnp_max_c_kernel<_DataType>>(DPNP_QUEUE);

            _DataType* res = std::max_element(policy, array_1, array_1 + size);
            policy.queue().wait();

            result[0] = *res;
        }
    }
    else
    {
        size_t res_ndim = ndim - naxis;
        size_t res_shape[res_ndim];
        int ind = 0;
        for (size_t i = 0; i < ndim; i++)
        {
            bool found = false;
            for (size_t j = 0; j < naxis; j++)
            {
                if (axis[j] == i)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                res_shape[ind] = shape[i];
                ind++;
            }
        }

        size_t input_shape_offsets[ndim];
        size_t acc = 1;
        for (size_t i = ndim - 1; i > 0; --i)
        {
            input_shape_offsets[i] = acc;
            acc *= shape[i];
        }
        input_shape_offsets[0] = acc;

        size_t output_shape_offsets[res_ndim];
        acc = 1;
        if (res_ndim > 0)
        {
            for (size_t i = res_ndim - 1; i > 0; --i)
            {
                output_shape_offsets[i] = acc;
                acc *= res_shape[i];
            }
        }
        output_shape_offsets[0] = acc;

        size_t size_result = 1;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            size_result *= res_shape[i];
        }

        //init result array
        for (size_t result_idx = 0; result_idx < size_result; ++result_idx)
        {
            size_t xyz[res_ndim];
            size_t remainder = result_idx;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                xyz[i] = remainder / output_shape_offsets[i];
                remainder = remainder - xyz[i] * output_shape_offsets[i];
            }

            size_t source_axis[ndim];
            size_t result_axis_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                bool found = false;
                for (size_t i = 0; i < naxis; ++i)
                {
                    if (axis[i] == idx)
                    {
                        found = true;
                        break;
                    }
                }
                if (found)
                {
                    source_axis[idx] = 0;
                }
                else
                {
                    source_axis[idx] = xyz[result_axis_idx];
                    result_axis_idx++;
                }
            }

            size_t source_idx = 0;
            for (size_t i = 0; i < ndim; ++i)
            {
                source_idx += input_shape_offsets[i] * source_axis[i];
            }

            result[result_idx] = array_1[source_idx];
        }

        for (size_t source_idx = 0; source_idx < size_input; ++source_idx)
        {
            // reconstruct x,y,z from linear source_idx
            size_t xyz[ndim];
            size_t remainder = source_idx;
            for (size_t i = 0; i < ndim; ++i)
            {
                xyz[i] = remainder / input_shape_offsets[i];
                remainder = remainder - xyz[i] * input_shape_offsets[i];
            }

            // extract result axis
            size_t result_axis[res_ndim];
            size_t result_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                // try to find current idx in axis array
                bool found = false;
                for (size_t i = 0; i < naxis; ++i)
                {
                    if (axis[i] == idx)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    result_axis[result_idx] = xyz[idx];
                    result_idx++;
                }
            }

            // Construct result offset
            size_t result_offset = 0;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                result_offset += output_shape_offsets[i] * result_axis[i];
            }

            if (result[result_offset] < array_1[source_idx])
            {
                result[result_offset] = array_1[source_idx];
            }
        }
    }

    return;
}

template <typename _DataType, typename _ResultType>
void dpnp_mean_c(void* array1_in,
                 void* result1,
                 const shape_elem_type* shape,
                 size_t ndim,
                 const shape_elem_type* axis,
                 size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    const size_t size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size, true);
    DPNPC_ptr_adapter<_ResultType> result_ptr(result1, 1, true, true);
    _DataType* array = input1_ptr.get_ptr();
    _ResultType* result = result_ptr.get_ptr();

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major /*, _ResultType*/>(1, size, array);

        sycl::event event = mkl_stats::mean(DPNP_QUEUE, dataset, result);

        event.wait();
    }
    else
    {
        _ResultType* sum = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(1 * sizeof(_ResultType)));

        dpnp_sum_c<_ResultType, _DataType>(sum, array, shape, ndim, axis, naxis, nullptr, nullptr);

        result[0] = sum[0] / static_cast<_ResultType>(size);

        dpnp_memory_free_c(sum);
    }

    return;
}

template <typename _DataType, typename _ResultType>
void dpnp_median_c(void* array1_in,
                   void* result1,
                   const shape_elem_type* shape,
                   size_t ndim,
                   const shape_elem_type* axis,
                   size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    const size_t size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!size)
    {
        return;
    }

    DPNPC_ptr_adapter<_ResultType> result_ptr(result1, 1, true, true);
    _ResultType* result = result_ptr.get_ptr();

    _DataType* sorted = reinterpret_cast<_DataType*>(dpnp_memory_alloc_c(size * sizeof(_DataType)));

    dpnp_sort_c<_DataType>(array1_in, sorted, size);

    if (size % 2 == 0)
    {
        result[0] = static_cast<_ResultType>(sorted[size / 2] + sorted[size / 2 - 1]) / 2;
    }
    else
    {
        result[0] = sorted[(size - 1) / 2];
    }

    dpnp_memory_free_c(sorted);

    return;
}

template <typename _DataType>
class dpnp_min_c_kernel;

template <typename _DataType>
void dpnp_min_c(void* array1_in,
                void* result1,
                const size_t result_size,
                const shape_elem_type* shape,
                size_t ndim,
                const shape_elem_type* axis,
                size_t naxis)
{
    __attribute__((unused)) void* tmp = (void*)(axis + naxis);

    const size_t size_input = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!size_input)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size_input, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, result_size, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    if (naxis == 0)
    {
        if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
        {
            // Required initializing the result before call the function
            result[0] = array_1[0];

            auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, size_input, array_1);

            sycl::event event = mkl_stats::min(DPNP_QUEUE, dataset, result);

            event.wait();
        }
        else
        {
            auto policy = oneapi::dpl::execution::make_device_policy<class dpnp_min_c_kernel<_DataType>>(DPNP_QUEUE);

            _DataType* res = std::min_element(policy, array_1, array_1 + size_input);
            policy.queue().wait();

            result[0] = *res;
        }
    }
    else
    {
        size_t res_ndim = ndim - naxis;
        size_t res_shape[res_ndim];
        int ind = 0;
        for (size_t i = 0; i < ndim; i++)
        {
            bool found = false;
            for (size_t j = 0; j < naxis; j++)
            {
                if (axis[j] == i)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                res_shape[ind] = shape[i];
                ind++;
            }
        }

        size_t input_shape_offsets[ndim];
        size_t acc = 1;
        for (size_t i = ndim - 1; i > 0; --i)
        {
            input_shape_offsets[i] = acc;
            acc *= shape[i];
        }
        input_shape_offsets[0] = acc;

        size_t output_shape_offsets[res_ndim];
        acc = 1;
        if (res_ndim > 0)
        {
            for (size_t i = res_ndim - 1; i > 0; --i)
            {
                output_shape_offsets[i] = acc;
                acc *= res_shape[i];
            }
        }
        output_shape_offsets[0] = acc;

        size_t size_result = 1;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            size_result *= res_shape[i];
        }

        //init result array
        for (size_t result_idx = 0; result_idx < size_result; ++result_idx)
        {
            size_t xyz[res_ndim];
            size_t remainder = result_idx;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                xyz[i] = remainder / output_shape_offsets[i];
                remainder = remainder - xyz[i] * output_shape_offsets[i];
            }

            size_t source_axis[ndim];
            size_t result_axis_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                bool found = false;
                for (size_t i = 0; i < naxis; ++i)
                {
                    if (axis[i] == idx)
                    {
                        found = true;
                        break;
                    }
                }
                if (found)
                {
                    source_axis[idx] = 0;
                }
                else
                {
                    source_axis[idx] = xyz[result_axis_idx];
                    result_axis_idx++;
                }
            }

            size_t source_idx = 0;
            for (size_t i = 0; i < ndim; ++i)
            {
                source_idx += input_shape_offsets[i] * source_axis[i];
            }

            result[result_idx] = array_1[source_idx];
        }

        for (size_t source_idx = 0; source_idx < size_input; ++source_idx)
        {
            // reconstruct x,y,z from linear source_idx
            size_t xyz[ndim];
            size_t remainder = source_idx;
            for (size_t i = 0; i < ndim; ++i)
            {
                xyz[i] = remainder / input_shape_offsets[i];
                remainder = remainder - xyz[i] * input_shape_offsets[i];
            }

            // extract result axis
            size_t result_axis[res_ndim];
            size_t result_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                // try to find current idx in axis array
                bool found = false;
                for (size_t i = 0; i < naxis; ++i)
                {
                    if (axis[i] == idx)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    result_axis[result_idx] = xyz[idx];
                    result_idx++;
                }
            }

            // Construct result offset
            size_t result_offset = 0;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                result_offset += output_shape_offsets[i] * result_axis[i];
            }

            if (result[result_offset] > array_1[source_idx])
            {
                result[result_offset] = array_1[source_idx];
            }
        }
    }

    return;
}

template <typename _DataType>
void dpnp_nanvar_c(void* array1_in, void* mask_arr1, void* result1, const size_t result_size, size_t arr_size)
{
    if ((array1_in == nullptr) || (mask_arr1 == nullptr) || (result1 == nullptr))
    {
        return;
    }

    if (arr_size == 0)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, arr_size, true);
    DPNPC_ptr_adapter<bool> input2_ptr(mask_arr1, arr_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, result_size, true, true);
    _DataType* array1 = input1_ptr.get_ptr();
    bool* mask_arr = input2_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

    size_t ind = 0;
    for (size_t i = 0; i < arr_size; ++i)
    {
        if (!mask_arr[i])
        {
            result[ind] = array1[i];
            ind += 1;
        }
    }

    return;
}

template <typename _DataType, typename _ResultType>
void dpnp_std_c(void* array1_in,
                void* result1,
                const shape_elem_type* shape,
                size_t ndim,
                const shape_elem_type* axis,
                size_t naxis,
                size_t ddof)
{
    _ResultType* var = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(1 * sizeof(_ResultType)));

    dpnp_var_c<_DataType, _ResultType>(array1_in, var, shape, ndim, axis, naxis, ddof);

    const size_t result1_size = 1;
    const size_t result1_ndim = 1;
    const size_t result1_shape_size_in_bytes = result1_ndim * sizeof(shape_elem_type);
    const size_t result1_strides_size_in_bytes = result1_ndim * sizeof(shape_elem_type);
    shape_elem_type* result1_shape =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(result1_shape_size_in_bytes));
    *result1_shape = 1;
    shape_elem_type* result1_strides =
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(result1_strides_size_in_bytes));
    *result1_strides = 1;

    const size_t var_size = 1;
    const size_t var_ndim = 1;
    const size_t var_shape_size_in_bytes = var_ndim * sizeof(shape_elem_type);
    const size_t var_strides_size_in_bytes = var_ndim * sizeof(shape_elem_type);
    shape_elem_type* var_shape = reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(var_shape_size_in_bytes));
    *var_shape = 1;
    shape_elem_type* var_strides = reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(var_strides_size_in_bytes));
    *var_strides = 1;

    dpnp_sqrt_c<_ResultType, _ResultType>(result1,
                                          result1_size,
                                          result1_ndim,
                                          result1_shape,
                                          result1_strides,
                                          var,
                                          var_size,
                                          var_ndim,
                                          var_shape,
                                          var_strides,
                                          NULL);

    dpnp_memory_free_c(var);
    dpnp_memory_free_c(result1_shape);
    dpnp_memory_free_c(result1_strides);
    dpnp_memory_free_c(var_shape);
    dpnp_memory_free_c(var_strides);

    return;
}

template <typename _DataType, typename _ResultType>
class dpnp_var_c_kernel;

template <typename _DataType, typename _ResultType>
void dpnp_var_c(void* array1_in,
                void* result1,
                const shape_elem_type* shape,
                size_t ndim,
                const shape_elem_type* axis,
                size_t naxis,
                size_t ddof)
{
    const size_t size = std::accumulate(shape, shape + ndim, 1, std::multiplies<shape_elem_type>());
    if (!size)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    DPNPC_ptr_adapter<_ResultType> result_ptr(result1, 1, true, true);
    _DataType* array1 = input1_ptr.get_ptr();
    _ResultType* result = result_ptr.get_ptr();

    _ResultType* mean = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(1 * sizeof(_ResultType)));
    dpnp_mean_c<_DataType, _ResultType>(array1, mean, shape, ndim, axis, naxis);
    _ResultType mean_val = mean[0];

    _ResultType* squared_deviations = reinterpret_cast<_ResultType*>(dpnp_memory_alloc_c(size * sizeof(_ResultType)));

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
        {
            _ResultType deviation = static_cast<_ResultType>(array1[i]) - mean_val;
            squared_deviations[i] = deviation * deviation;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_var_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    dpnp_mean_c<_ResultType, _ResultType>(squared_deviations, mean, shape, ndim, axis, naxis);
    mean_val = mean[0];

    result[0] = mean_val * size / static_cast<_ResultType>(size - ddof);

    dpnp_memory_free_c(mean);
    dpnp_memory_free_c(squared_deviations);

    return;
}

void func_map_init_statistics(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_INT] = {eft_INT,
                                                               (void*)dpnp_correlate_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_LNG] = {eft_LNG,
                                                               (void*)dpnp_correlate_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_FLT] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_INT][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_INT] = {eft_LNG,
                                                               (void*)dpnp_correlate_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_LNG] = {eft_LNG,
                                                               (void*)dpnp_correlate_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_FLT] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_LNG][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_INT] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_LNG] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_correlate_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_correlate_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_INT] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_LNG] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_correlate_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_CORRELATE][eft_DBL][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_correlate_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COUNT_NONZERO][eft_BLN][eft_BLN] = {eft_LNG, (void*)dpnp_count_nonzero_c<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COUNT_NONZERO][eft_INT][eft_INT] = {eft_LNG,
                                                                   (void*)dpnp_count_nonzero_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COUNT_NONZERO][eft_LNG][eft_LNG] = {eft_LNG,
                                                                   (void*)dpnp_count_nonzero_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COUNT_NONZERO][eft_FLT][eft_FLT] = {eft_LNG,
                                                                   (void*)dpnp_count_nonzero_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COUNT_NONZERO][eft_DBL][eft_DBL] = {eft_LNG,
                                                                   (void*)dpnp_count_nonzero_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_COV][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_FLT][eft_FLT] = {eft_DBL, (void*)dpnp_cov_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COV][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cov_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MAX][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_max_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_max_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_max_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MAX][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_max_c<double>};

    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_mean_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_mean_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_mean_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MEAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_mean_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_median_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_median_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_median_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MEDIAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_median_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MIN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_min_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_min_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_min_c<float>};
    fmap[DPNPFuncName::DPNP_FN_MIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_min_c<double>};

    fmap[DPNPFuncName::DPNP_FN_NANVAR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_nanvar_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_NANVAR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_nanvar_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_NANVAR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_nanvar_c<float>};
    fmap[DPNPFuncName::DPNP_FN_NANVAR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_nanvar_c<double>};

    fmap[DPNPFuncName::DPNP_FN_STD][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_std_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_std_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_std_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_STD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_std_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_VAR][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_var_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_var_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_var_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_VAR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_var_c<double, double>};

    return;
}
