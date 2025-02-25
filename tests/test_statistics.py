import pytest

import dpnp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("size",
                         [2, 4, 8, 16, 3, 9, 27, 81])
def test_median(type, size):
    a = numpy.arange(size, dtype=type)
    ia = dpnp.array(a)

    np_res = numpy.median(a)
    dpnp_res = dpnp.median(ia)

    numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("axis",
                         [0, 1, -1, 2, -2, (1, 2), (0, -2)])
def test_max(axis):
    a = numpy.arange(768, dtype=numpy.float64).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = numpy.max(a, axis=axis)
    dpnp_res = dpnp.max(ia, axis=axis)

    numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("array",
                         [[2, 0, 6, 2],
                          [2, 0, 6, 2, 5, 6, 7, 8],
                          [],
                          [2, 1, numpy.nan, 5, 3],
                          [-1, numpy.nan, 1, numpy.inf],
                          [3, 6, 0, 1],
                          [3, 6, 0, 1, 8],
                          [3, 2, 9, 6, numpy.nan],
                          [numpy.nan, numpy.nan, numpy.inf, numpy.nan],
                          [[2, 0], [6, 2]],
                          [[2, 0, 6, 2], [5, 6, 7, 8]],
                          [[[2, 0], [6, 2]], [[5, 6], [7, 8]]],
                          [[-1, numpy.nan], [1, numpy.inf]],
                          [[numpy.nan, numpy.nan], [numpy.inf, numpy.nan]]],
                         ids=['[2, 0, 6, 2]',
                              '[2, 0, 6, 2, 5, 6, 7, 8]',
                              '[]',
                              '[2, 1, np.nan, 5, 3]',
                              '[-1, np.nan, 1, np.inf]',
                              '[3, 6, 0, 1]',
                              '[3, 6, 0, 1, 8]',
                              '[3, 2, 9, 6, np.nan]',
                              '[np.nan, np.nan, np.inf, np.nan]',
                              '[[2, 0], [6, 2]]',
                              '[[2, 0, 6, 2], [5, 6, 7, 8]]',
                              '[[[2, 0], [6, 2]], [[5, 6], [7, 8]]]',
                              '[[-1, np.nan], [1, np.inf]]',
                              '[[np.nan, np.nan], [np.inf, np.nan]]'])
def test_nanvar(array):
    a = numpy.array(array)
    ia = dpnp.array(a)
    for ddof in range(a.ndim):
        expected = numpy.nanvar(a, ddof=ddof)
        result = dpnp.nanvar(ia, ddof=ddof)
        numpy.testing.assert_array_equal(expected, result)

    expected = numpy.nanvar(a, axis=None, ddof=0)
    result = dpnp.nanvar(ia, axis=None, ddof=0)
    numpy.testing.assert_array_equal(expected, result)


class TestBincount:

    @pytest.mark.parametrize("array",
                             [[1, 2, 3],
                              [1, 2, 2, 1, 2, 4],
                              [2, 2, 2, 2]],
                             ids=['[1, 2, 3]',
                                  '[1, 2, 2, 1, 2, 4]',
                                  '[2, 2, 2, 2]'])
    @pytest.mark.parametrize("minlength",
                             [0, 1, 3, 5],
                             ids=['0', '1', '3', '5'])
    def test_bincount_minlength(self, array, minlength):
        np_a = numpy.array(array)
        dpnp_a = dpnp.array(array)

        expected = numpy.bincount(np_a, minlength=minlength)
        result = dpnp.bincount(dpnp_a, minlength=minlength)
        numpy.testing.assert_array_equal(expected, result)

    @pytest.mark.parametrize("array",
                             [[1, 2, 2, 1, 2, 4]],
                             ids=['[1, 2, 2, 1, 2, 4]'])
    @pytest.mark.parametrize("weights",
                             [None,
                              [0.3, 0.5, 0.2, 0.7, 1., -0.6],
                              [2, 2, 2, 2, 2, 2]],
                             ids=['None',
                                  '[0.3, 0.5, 0.2, 0.7, 1., -0.6]',
                                  '[2, 2, 2, 2, 2, 2]'])
    def test_bincount_weights(self, array, weights):
        np_a = numpy.array(array)
        dpnp_a = dpnp.array(array)

        expected = numpy.bincount(np_a, weights=weights)
        result = dpnp.bincount(dpnp_a, weights=weights)
        numpy.testing.assert_array_equal(expected, result)
