# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

from libcpp cimport bool as cpp_bool

from dpnp.dpnp_utils.dpnp_algo_utils cimport dpnp_descriptor

from dpnp.dpnp_algo cimport shape_elem_type, shape_type_c

cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_ABSOLUTE
        DPNP_FN_ADD
        DPNP_FN_ALL
        DPNP_FN_ALLCLOSE
        DPNP_FN_ANY
        DPNP_FN_ARANGE
        DPNP_FN_ARCCOS
        DPNP_FN_ARCCOSH
        DPNP_FN_ARCSIN
        DPNP_FN_ARCSINH
        DPNP_FN_ARCTAN
        DPNP_FN_ARCTAN2
        DPNP_FN_ARCTANH
        DPNP_FN_ARGMAX
        DPNP_FN_ARGMIN
        DPNP_FN_ARGSORT
        DPNP_FN_AROUND
        DPNP_FN_ASTYPE
        DPNP_FN_BITWISE_AND
        DPNP_FN_BITWISE_OR
        DPNP_FN_BITWISE_XOR
        DPNP_FN_CBRT
        DPNP_FN_CEIL
        DPNP_FN_CHOLESKY
        DPNP_FN_CHOOSE
        DPNP_FN_CONJIGUATE
        DPNP_FN_COPY
        DPNP_FN_COPYSIGN
        DPNP_FN_COPYTO
        DPNP_FN_CORRELATE
        DPNP_FN_COS
        DPNP_FN_COSH
        DPNP_FN_COV
        DPNP_FN_COUNT_NONZERO
        DPNP_FN_CROSS
        DPNP_FN_CUMPROD
        DPNP_FN_CUMSUM
        DPNP_FN_DEGREES
        DPNP_FN_DET
        DPNP_FN_DIAG
        DPNP_FN_DIAG_INDICES
        DPNP_FN_DIAGONAL
        DPNP_FN_DIVIDE
        DPNP_FN_DOT
        DPNP_FN_EDIFF1D
        DPNP_FN_EIG
        DPNP_FN_EIGVALS
        DPNP_FN_ERF
        DPNP_FN_EYE
        DPNP_FN_EXP
        DPNP_FN_EXP2
        DPNP_FN_EXPM1
        DPNP_FN_FABS
        DPNP_FN_FFT_FFT
        DPNP_FN_FILL_DIAGONAL
        DPNP_FN_FLATTEN
        DPNP_FN_FLOOR
        DPNP_FN_FLOOR_DIVIDE
        DPNP_FN_FMOD
        DPNP_FN_FULL
        DPNP_FN_FULL_LIKE
        DPNP_FN_HYPOT
        DPNP_FN_IDENTITY
        DPNP_FN_INITVAL
        DPNP_FN_INV
        DPNP_FN_INVERT
        DPNP_FN_KRON
        DPNP_FN_LEFT_SHIFT
        DPNP_FN_LOG
        DPNP_FN_LOG10
        DPNP_FN_LOG1P
        DPNP_FN_LOG2
        DPNP_FN_MATMUL
        DPNP_FN_MATRIX_RANK
        DPNP_FN_MAX
        DPNP_FN_MAXIMUM
        DPNP_FN_MEAN
        DPNP_FN_MEDIAN
        DPNP_FN_MIN
        DPNP_FN_MINIMUM
        DPNP_FN_MODF
        DPNP_FN_MULTIPLY
        DPNP_FN_NANVAR
        DPNP_FN_NEGATIVE
        DPNP_FN_NONZERO
        DPNP_FN_ONES
        DPNP_FN_ONES_LIKE
        DPNP_FN_PARTITION
        DPNP_FN_PLACE
        DPNP_FN_POWER
        DPNP_FN_PROD
        DPNP_FN_PTP
        DPNP_FN_PUT
        DPNP_FN_QR
        DPNP_FN_RADIANS
        DPNP_FN_REMAINDER
        DPNP_FN_RECIP
        DPNP_FN_REPEAT
        DPNP_FN_RIGHT_SHIFT
        DPNP_FN_RNG_BETA
        DPNP_FN_RNG_BINOMIAL
        DPNP_FN_RNG_CHISQUARE
        DPNP_FN_RNG_EXPONENTIAL
        DPNP_FN_RNG_F
        DPNP_FN_RNG_GAMMA
        DPNP_FN_RNG_GAUSSIAN
        DPNP_FN_RNG_GEOMETRIC
        DPNP_FN_RNG_GUMBEL
        DPNP_FN_RNG_HYPERGEOMETRIC
        DPNP_FN_RNG_LAPLACE
        DPNP_FN_RNG_LOGISTIC
        DPNP_FN_RNG_LOGNORMAL
        DPNP_FN_RNG_MULTINOMIAL
        DPNP_FN_RNG_MULTIVARIATE_NORMAL
        DPNP_FN_RNG_NEGATIVE_BINOMIAL
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE
        DPNP_FN_RNG_NORMAL
        DPNP_FN_RNG_PARETO
        DPNP_FN_RNG_POISSON
        DPNP_FN_RNG_POWER
        DPNP_FN_PUT_ALONG_AXIS
        DPNP_FN_RNG_RAYLEIGH
        DPNP_FN_RNG_SHUFFLE
        DPNP_FN_RNG_SRAND
        DPNP_FN_RNG_STANDARD_CAUCHY
        DPNP_FN_RNG_STANDARD_EXPONENTIAL
        DPNP_FN_RNG_STANDARD_GAMMA
        DPNP_FN_RNG_STANDARD_NORMAL
        DPNP_FN_RNG_STANDARD_T
        DPNP_FN_RNG_TRIANGULAR
        DPNP_FN_RNG_UNIFORM
        DPNP_FN_RNG_VONMISES
        DPNP_FN_RNG_WALD
        DPNP_FN_RNG_WEIBULL
        DPNP_FN_RNG_ZIPF
        DPNP_FN_SEARCHSORTED
        DPNP_FN_SIGN
        DPNP_FN_SIN
        DPNP_FN_SINH
        DPNP_FN_SORT
        DPNP_FN_SQRT
        DPNP_FN_SQUARE
        DPNP_FN_STD
        DPNP_FN_SUBTRACT
        DPNP_FN_SUM
        DPNP_FN_SVD
        DPNP_FN_TAKE
        DPNP_FN_TAN
        DPNP_FN_TANH
        DPNP_FN_TRACE
        DPNP_FN_TRANSPOSE
        DPNP_FN_TRAPZ
        DPNP_FN_TRI
        DPNP_FN_TRIL
        DPNP_FN_TRIU
        DPNP_FN_TRUNC
        DPNP_FN_VANDER
        DPNP_FN_VAR
        DPNP_FN_ZEROS
        DPNP_FN_ZEROS_LIKE

cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE
        DPNP_FT_CMPLX64
        DPNP_FT_CMPLX128
        DPNP_FT_BOOL

cdef extern from "dpnp_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type) except +


cdef extern from "dpnp_iface.hpp" namespace "QueueOptions":  # need this namespace for Enum import
    cdef enum QueueOptions "QueueOptions":
        CPU_SELECTOR
        GPU_SELECTOR
        AUTO_SELECTOR

cdef extern from "constants.hpp":
    void dpnp_python_constants_initialize_c(void * py_none, void * py_nan)

cdef extern from "dpnp_iface.hpp":
    void dpnp_queue_initialize_c(QueueOptions selector)
    size_t dpnp_queue_is_cpu_c()

    char * dpnp_memory_alloc_c(size_t size_in_bytes) except +
    void dpnp_memory_free_c(void * ptr)
    void dpnp_memory_memcpy_c(void * dst, const void * src, size_t size_in_bytes)
    void dpnp_rng_srand_c(size_t seed)


# C function pointer to the C library template functions
ctypedef void(*fptr_1out_t)(void * , size_t)
ctypedef void(*fptr_1in_1out_t)(void *, void * , size_t)
ctypedef void(*fptr_1in_1out_strides_t)(void *, const size_t, const size_t,
                                        const shape_elem_type * , const shape_elem_type * ,
                                        void *, const size_t, const size_t,
                                        const shape_elem_type * , const shape_elem_type * ,
                                        const long * )
ctypedef void(*fptr_2in_1out_t)(void * , const void * , const size_t, const shape_elem_type * , const size_t,
                                const void *, const size_t, const shape_elem_type * , const size_t, const long * )
ctypedef void(*fptr_2in_1out_strides_t)(void *, const size_t, const size_t,
                                        const shape_elem_type * , const shape_elem_type * ,
                                        void *, const size_t, const size_t,
                                        const shape_elem_type * , const shape_elem_type * ,
                                        void *, const size_t, const size_t,
                                        const shape_elem_type * , const shape_elem_type * ,
                                        const long * )
ctypedef void(*fptr_blas_gemm_2in_1out_t)(void *, void * , void * , size_t, size_t, size_t)
ctypedef void(*dpnp_reduction_c_t)(void *, const void * , const shape_elem_type*, const size_t, const shape_elem_type*, const size_t, const void * , const long*)

cpdef dpnp_descriptor dpnp_astype(dpnp_descriptor x1, dtype)
cpdef dpnp_descriptor dpnp_flatten(dpnp_descriptor x1)


"""
Internal functions
"""
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype)
cdef dpnp_DPNPFuncType_to_dtype(size_t type)


"""
Bitwise functions
"""
cpdef dpnp_descriptor dpnp_bitwise_and(dpnp_descriptor x1_obj,
                                       dpnp_descriptor x2_obj,
                                       object dtype=*,
                                       dpnp_descriptor out=*,
                                       object where=*)
cpdef dpnp_descriptor dpnp_bitwise_or(dpnp_descriptor x1_obj,
                                      dpnp_descriptor x2_obj,
                                      object dtype=*,
                                      dpnp_descriptor out=*,
                                      object where=*)
cpdef dpnp_descriptor dpnp_bitwise_xor(dpnp_descriptor x1_obj,
                                       dpnp_descriptor x2_obj,
                                       object dtype=*,
                                       dpnp_descriptor out=*,
                                       object where=*)
cpdef dpnp_descriptor dpnp_invert(dpnp_descriptor x1)
cpdef dpnp_descriptor dpnp_left_shift(dpnp_descriptor x1_obj,
                                      dpnp_descriptor x2_obj,
                                      object dtype=*,
                                      dpnp_descriptor out=*,
                                      object where=*)
cpdef dpnp_descriptor dpnp_right_shift(dpnp_descriptor x1_obj,
                                       dpnp_descriptor x2_obj,
                                       object dtype=*,
                                       dpnp_descriptor out=*,
                                       object where=*)


"""
Logic functions
"""
cpdef dpnp_descriptor dpnp_equal(dpnp_descriptor array1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_greater(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_greater_equal(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_isclose(dpnp_descriptor input1, dpnp_descriptor input2,
                                   double rtol=*, double atol=*, cpp_bool equal_nan=*)
cpdef dpnp_descriptor dpnp_less(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_less_equal(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_logical_and(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_logical_not(dpnp_descriptor input1)
cpdef dpnp_descriptor dpnp_logical_or(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_logical_xor(dpnp_descriptor input1, dpnp_descriptor input2)
cpdef dpnp_descriptor dpnp_not_equal(dpnp_descriptor input1, dpnp_descriptor input2)


"""
Linear algebra
"""
cpdef dpnp_descriptor dpnp_dot(dpnp_descriptor in_array1, dpnp_descriptor in_array2)
cpdef dpnp_descriptor dpnp_matmul(dpnp_descriptor in_array1, dpnp_descriptor in_array2, dpnp_descriptor out=*)


"""
Array creation routines
"""
cpdef dpnp_descriptor dpnp_arange(start, stop, step, dtype)
cpdef dpnp_descriptor dpnp_init_val(shape, dtype, value)
cpdef dpnp_descriptor dpnp_full(result_shape, value_in, result_dtype)  # same as dpnp_init_val
cpdef dpnp_descriptor dpnp_copy(dpnp_descriptor x1)

"""
Mathematical functions
"""
cpdef dpnp_descriptor dpnp_add(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                               dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_arctan2(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                   dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_divide(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                  dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_hypot(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                 dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_maximum(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                   dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_minimum(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                   dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_multiply(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                    dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_negative(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_power(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                 dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_remainder(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                     dpnp_descriptor out=*, object where=*)
cpdef dpnp_descriptor dpnp_subtract(dpnp_descriptor x1_obj, dpnp_descriptor x2_obj, object dtype=*,
                                    dpnp_descriptor out=*, object where=*)


"""
Array manipulation routines
"""
cpdef dpnp_descriptor dpnp_repeat(dpnp_descriptor array1, repeats, axes=*)
cpdef dpnp_descriptor dpnp_transpose(dpnp_descriptor array1, axes=*)


"""
Statistics functions
"""
cpdef dpnp_descriptor dpnp_cov(dpnp_descriptor array1)
cpdef object dpnp_mean(dpnp_descriptor a, axis)
cpdef dpnp_descriptor dpnp_min(dpnp_descriptor a, axis)


"""
Sorting functions
"""
cpdef dpnp_descriptor dpnp_argsort(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_sort(dpnp_descriptor array1)

"""
Searching functions
"""
cpdef dpnp_descriptor dpnp_argmax(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_argmin(dpnp_descriptor array1)

"""
Trigonometric functions
"""
cpdef dpnp_descriptor dpnp_arccos(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_arccosh(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_arcsin(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_arcsinh(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_arctan(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_arctanh(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_cbrt(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_cos(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_cosh(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_degrees(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_exp(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_exp2(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_expm1(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_log(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_log10(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_log1p(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_log2(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_radians(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_recip(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_sin(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_sinh(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_sqrt(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_square(dpnp_descriptor array1)
cpdef dpnp_descriptor dpnp_tan(dpnp_descriptor array1, dpnp_descriptor out)
cpdef dpnp_descriptor dpnp_tanh(dpnp_descriptor array1)
