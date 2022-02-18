import pytest

import dpnp
import dpctl
import numpy


list_of_backend_str = [
    "host",
    "level_zero",
    "opencl",
]

list_of_device_type_str = [
    "host",
    "gpu",
    "cpu",
]

available_devices = dpctl.get_devices()

valid_devices = []
for device in available_devices:
    if device.backend.name not in list_of_backend_str:
        pass
    elif device.device_type.name not in list_of_device_type_str:
        pass
    else:
        valid_devices.append(device)


def assert_sycl_queue_equal(result, expected):
    exec_queue = dpctl.utils.get_execution_queue([result, expected])
    assert exec_queue is not None


def vvsort(val, vec, size, xp):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            unravel_imax = numpy.unravel_index(imax, val.shape)
            unravel_j = numpy.unravel_index(j, val.shape)
            if xp.abs(val[unravel_imax]) < xp.abs(val[unravel_j]):
                imax = j

        unravel_i = numpy.unravel_index(i, val.shape)
        unravel_imax = numpy.unravel_index(imax, val.shape)

        temp = xp.empty(tuple(), dtype=vec.dtype)
        temp[()] = val[unravel_i]  # make a copy
        val[unravel_i] = val[unravel_imax]
        val[unravel_imax] = temp

        for k in range(size):
            temp = xp.empty(tuple(), dtype=val.dtype)
            temp[()] = vec[k, i]  # make a copy
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_matmul(device):
    data1 = [[1., 1., 1.], [1., 1., 1.]]
    data2 = [[1., 1.], [1., 1.], [1., 1.]]

    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = numpy.matmul(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = dpnp.matmul(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("func",
                         [])
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_2in_1out(func, device):
    data1 = [1., 1., 1., 1., 1.]
    data2 = [1., 2., 3., 4., 5.]

    x1_orig = numpy.array(data1)
    x2_orig = numpy.array(data2)
    expected = getattr(numpy, func)(x1_orig, x2_orig)

    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    numpy.testing.assert_array_equal(result, expected)

    expected_queue = x1.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("type", ['complex128'])
@pytest.mark.parametrize("device",
                         valid_devices,
                         ids=[device.filter_string for device in valid_devices])
def test_fft(type, device):
    data = numpy.arange(100, dtype=numpy.dtype(type))

    dpnp_data = dpnp.array(data, device=device)

    expected = numpy.fft.fft(data)
    result = dpnp.fft.fft(dpnp_data)

    numpy.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-7)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_cholesky(device):
    data = [[[1., -2.], [2., 5.]], [[1., -2.], [2., 5.]]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.cholesky(dpnp_data)
    expected = numpy.linalg.cholesky(numpy_data)
    numpy.testing.assert_array_equal(expected, result)

    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_det(device):
    data = [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.det(dpnp_data)
    expected = numpy.linalg.det(numpy_data)
    numpy.testing.assert_allclose(expected, result)
    
    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_eig(device):
    size = 4
    a = numpy.arange(size * size, dtype='float64').reshape((size, size))
    symm_orig = numpy.tril(a) + numpy.tril(a, -1).T + numpy.diag(numpy.full((size,), size * size, dtype='float64'))
    numpy_data = symm_orig
    dpnp_symm_orig = dpnp.array(numpy_data, device=device)
    dpnp_data = dpnp_symm_orig

    dpnp_val, dpnp_vec = dpnp.linalg.eig(dpnp_data)
    numpy_val, numpy_vec = numpy.linalg.eig(numpy_data)
    
    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size, dpnp)

    # NP sort val/vec by abs value
    vvsort(numpy_val, numpy_vec, size, numpy)

    # NP change sign of vectors
    for i in range(numpy_vec.shape[1]):
        if numpy_vec[0, i] * dpnp_vec[0, i] < 0:
            numpy_vec[:, i] = -numpy_vec[:, i]

    numpy.testing.assert_allclose(dpnp_val, numpy_val, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(dpnp_vec, numpy_vec, rtol=1e-05, atol=1e-05)
    
    assert (dpnp_val.dtype == numpy_val.dtype)
    assert (dpnp_vec.dtype == numpy_vec.dtype)
    assert (dpnp_val.shape == numpy_val.shape)
    assert (dpnp_vec.shape == numpy_vec.shape)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_val_queue = dpnp_val.get_array().sycl_queue
    dpnp_vec_queue = dpnp_vec.get_array().sycl_queue

    # compare queue and device    
    assert_sycl_queue_equal(dpnp_val_queue, expected_queue)
    assert dpnp_val_queue.sycl_device == expected_queue.sycl_device

    assert_sycl_queue_equal(dpnp_vec_queue, expected_queue)
    assert dpnp_vec_queue.sycl_device == expected_queue.sycl_device

    
@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_eigvals(device):
    data = [[0, 0], [0, 0]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.eigvals(dpnp_data)
    expected = numpy.linalg.eigvals(numpy_data)
    numpy.testing.assert_allclose(expected, result, atol=0.5)
    
    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_inv(device):
    data = [[1., 2.], [3., 4.]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.inv(dpnp_data)
    expected = numpy.linalg.inv(numpy_data)
    numpy.testing.assert_allclose(expected, result)
    
    expected_queue = dpnp_data.get_array().sycl_queue
    result_queue = result.get_array().sycl_queue

    assert_sycl_queue_equal(result_queue, expected_queue)
    assert result_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_matrix_rank(device):
    data = [[0, 0], [0, 0]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    result = dpnp.linalg.matrix_rank(dpnp_data)
    expected = numpy.linalg.matrix_rank(numpy_data)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("device",
                          valid_devices,
                          ids=[device.filter_string for device in valid_devices])
def test_qr(device):
    tol = 1e-11
    data = [[0, 1], [1, 1], [1, 1], [2, 1]]
    numpy_data = numpy.array(data)
    dpnp_data = dpnp.array(data, device=device)

    np_q, np_r = numpy.linalg.qr(numpy_data, "complete")
    dpnp_q, dpnp_r = dpnp.linalg.qr(dpnp_data, "complete")

    assert (dpnp_q.dtype == np_q.dtype)
    assert (dpnp_r.dtype == np_r.dtype)
    assert (dpnp_q.shape == np_q.shape)
    assert (dpnp_r.shape == np_r.shape)

    # NP change sign for comparison
    ncols = min(numpy_data.shape[0], numpy_data.shape[1])
    for i in range(ncols):
        j = numpy.where(numpy.abs(np_q[:, i]) > tol)[0][0]
        if np_q[j, i] * dpnp_q[j, i] < 0:
            np_q[:, i] = -np_q[:, i]
            np_r[i, :] = -np_r[i, :]

        if numpy.any(numpy.abs(np_r[i, :]) > tol):
            numpy.testing.assert_allclose(dpnp.asnumpy(dpnp_q)[:, i], np_q[:, i], rtol=tol, atol=tol)

    numpy.testing.assert_allclose(dpnp_r, np_r, rtol=tol, atol=tol)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_q_queue = dpnp_q.get_array().sycl_queue
    dpnp_r_queue = dpnp_r.get_array().sycl_queue

    # compare queue and device
    assert_sycl_queue_equal(dpnp_q_queue, expected_queue)
    assert dpnp_q_queue.sycl_device == expected_queue.sycl_device

    assert_sycl_queue_equal(dpnp_r_queue, expected_queue)
    assert dpnp_r_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("device",
                        valid_devices,
                        ids=[device.filter_string for device in valid_devices])
def test_svd(device):
    tol = 1e-12
    shape = (2,2)
    numpy_data = numpy.arange(shape[0] * shape[1]).reshape(shape)
    dpnp_data = dpnp.arange(shape[0] * shape[1]).reshape(shape)
    np_u, np_s, np_vt = numpy.linalg.svd(numpy_data)
    dpnp_u, dpnp_s, dpnp_vt = dpnp.linalg.svd(dpnp_data)


    print(type(dpnp_u))
    assert (dpnp_u.dtype == np_u.dtype)
    assert (dpnp_s.dtype == np_s.dtype)
    assert (dpnp_vt.dtype == np_vt.dtype)
    assert (dpnp_u.shape == np_u.shape)
    assert (dpnp_s.shape == np_s.shape)
    assert (dpnp_vt.shape == np_vt.shape)

    # check decomposition
    dpnp_diag_s = dpnp.zeros(shape, dtype=dpnp_s.dtype)
    for i in range(dpnp_s.size):
        dpnp_diag_s[i, i] = dpnp_s[i]

    # check decomposition
    numpy.testing.assert_allclose(dpnp_data, dpnp.dot(dpnp_u, dpnp.dot(dpnp_diag_s, dpnp_vt)), rtol=tol, atol=tol)

    for i in range(min(shape[0], shape[1])):
        if np_u[0, i] * dpnp_u[0, i] < 0:
            np_u[:, i] = -np_u[:, i]
            np_vt[i, :] = -np_vt[i, :]

    # compare vectors for non-zero values
    for i in range(numpy.count_nonzero(np_s > tol)):
        numpy.testing.assert_allclose(dpnp.asnumpy(dpnp_u)[:, i], np_u[:, i], rtol=tol, atol=tol)
        numpy.testing.assert_allclose(dpnp.asnumpy(dpnp_vt)[i, :], np_vt[i, :], rtol=tol, atol=tol)

    expected_queue = dpnp_data.get_array().sycl_queue
    dpnp_u_queue = dpnp_u.get_array().sycl_queue
    dpnp_s_queue = dpnp_s.get_array().sycl_queue
    dpnp_vt_queue = dpnp_vt.get_array().sycl_queue
 
    # compare queue and device
    assert_sycl_queue_equal(dpnp_u_queue, expected_queue)
    assert dpnp_u_queue.sycl_device == expected_queue.sycl_device

    assert_sycl_queue_equal(dpnp_s_queue, expected_queue)
    assert dpnp_s_queue.sycl_device == expected_queue.sycl_device

    assert_sycl_queue_equal(dpnp_vt_queue, expected_queue)
    assert dpnp_vt_queue.sycl_device == expected_queue.sycl_device


@pytest.mark.parametrize("usm_type",
                        ["host", "device", "shared"])
def test_uniform(usm_type):
    seed = 123
    low = 1.0
    high = 2.0
    res = dpnp.random.uniform(low, high, usm_type=usm_type)

    res_usm_type = res.get_array().usm_type
    assert usm_type == res_usm_type
