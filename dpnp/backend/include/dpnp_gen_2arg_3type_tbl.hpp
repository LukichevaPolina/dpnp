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

/*
 * This header file contains single argument element wise functions definitions
 *
 * Macro `MACRO_2ARG_3TYPES_OP` must be defined before usage
 *
 * Parameters:
 * - public name of the function and kernel name
 * - operation used to calculate the result
 * - mkl operation used to calculate the result
 *
 */

#ifndef MACRO_2ARG_3TYPES_OP
#error "MACRO_2ARG_3TYPES_OP is not defined"
#endif

#ifdef _SECTION_DOCUMENTATION_GENERATION_

#define MACRO_2ARG_3TYPES_OP(__name__, __operation1__, __operation2__)                                                  \
    /** @ingroup BACKEND_API                                                                                         */ \
    /** @brief Per element operation function __name__                                                               */ \
    /**                                                                                                              */ \
    /** Function "__name__" executes operator "__operation1__" over corresponding elements of input arrays           */ \
    /**                                                                                                              */ \
    /** @param[out] result_out      Output array.                                                                    */ \
    /** @param[in]  result_size     Output array size.                                                               */ \
    /** @param[in]  result_ndim     Number of output array dimensions.                                               */ \
    /** @param[in]  result_shape    Output array shape.                                                              */ \
    /** @param[in]  result_strides  Output array strides.                                                            */ \
    /** @param[in]  input1_in       Input array 1.                                                                   */ \
    /** @param[in]  input1_size     Input array 1 size.                                                              */ \
    /** @param[in]  input1_ndim     Number of input array 1 dimensions.                                              */ \
    /** @param[in]  input1_shape    Input array 1 shape.                                                             */ \
    /** @param[in]  input1_strides  Input array 1 strides.                                                           */ \
    /** @param[in]  input2_in       Input array 2.                                                                   */ \
    /** @param[in]  input2_size     Input array 2 size.                                                              */ \
    /** @param[in]  input2_ndim     Number of input array 2 dimensions.                                              */ \
    /** @param[in]  input2_shape    Input array 2 shape.                                                             */ \
    /** @param[in]  input2_strides  Input array 2 strides.                                                           */ \
    /** @param[in]  where           Where condition.                                                                 */ \
    template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>                          \
    void __name__(void* result_out,                                                                                     \
                  const size_t result_size,                                                                             \
                  const size_t result_ndim,                                                                             \
                  const shape_elem_type* result_shape,                                                                  \
                  const shape_elem_type* result_strides,                                                                \
                  const void* input1_in,                                                                                \
                  const size_t input1_size,                                                                             \
                  const size_t input1_ndim,                                                                             \
                  const shape_elem_type* input1_shape,                                                                  \
                  const shape_elem_type* input1_strides,                                                                \
                  const void* input2_in,                                                                                \
                  const size_t input2_size,                                                                             \
                  const size_t input2_ndim,                                                                             \
                  const shape_elem_type* input2_shape,                                                                  \
                  const shape_elem_type* input2_strides,                                                                \
                  const size_t* where)

#endif

MACRO_2ARG_3TYPES_OP(dpnp_add_c, input1_elem + input2_elem, oneapi::mkl::vm::add)
MACRO_2ARG_3TYPES_OP(dpnp_arctan2_c, sycl::atan2((double)input1_elem, (double)input2_elem), oneapi::mkl::vm::atan2)
MACRO_2ARG_3TYPES_OP(dpnp_copysign_c,
                     sycl::copysign((double)input1_elem, (double)input2_elem),
                     oneapi::mkl::vm::copysign)
MACRO_2ARG_3TYPES_OP(dpnp_divide_c, input1_elem / input2_elem, oneapi::mkl::vm::div)
MACRO_2ARG_3TYPES_OP(dpnp_fmod_c, sycl::fmod((double)input1_elem, (double)input2_elem), oneapi::mkl::vm::fmod)
MACRO_2ARG_3TYPES_OP(dpnp_hypot_c, sycl::hypot((double)input1_elem, (double)input2_elem), oneapi::mkl::vm::hypot)
MACRO_2ARG_3TYPES_OP(dpnp_maximum_c, sycl::max(input1_elem, input2_elem), oneapi::mkl::vm::fmax)
MACRO_2ARG_3TYPES_OP(dpnp_minimum_c, sycl::min(input1_elem, input2_elem), oneapi::mkl::vm::fmin)

// "multiply" needs to be standalone kernel (not autogenerated) due to complex algorithm. This is not an element wise.
// pytest "tests/third_party/cupy/creation_tests/test_ranges.py::TestMgrid::test_mgrid3"
// requires multiplication shape1[10] with shape2[10,1] and result expected as shape[10,10]
MACRO_2ARG_3TYPES_OP(dpnp_multiply_c, input1_elem* input2_elem, oneapi::mkl::vm::mul)

MACRO_2ARG_3TYPES_OP(dpnp_power_c, sycl::pow((double)input1_elem, (double)input2_elem), oneapi::mkl::vm::pow)
MACRO_2ARG_3TYPES_OP(dpnp_subtract_c, input1_elem - input2_elem, oneapi::mkl::vm::sub)

#undef MACRO_2ARG_3TYPES_OP
