#include <dpnp_iface.hpp>

#include <vector>
  
#include "gtest/gtest.h"

TEST (TestBackendRandomBeta, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    double a = 0.4;
    double b = 0.5;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_srand_c(seed);
        dpnp_rng_beta_c<double>(result1, a, b, size);

        dpnp_srand_c(seed);
        dpnp_rng_beta_c<double>(result2, a, b, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomNormal, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    double loc = 2.56;
    double scale = 0.8;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_srand_c(seed);
        dpnp_rng_normal_c<double>(result1, loc, scale, size);

        dpnp_srand_c(seed);
        dpnp_rng_normal_c<double>(result2, loc, scale, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

TEST (TestBackendRandomUniform, test_seed) {
    const size_t size = 256;
    size_t seed = 10;
    long low = 1;
    long high = 120;

    auto QueueOptionsDevices = std::vector<QueueOptions>{ QueueOptions::CPU_SELECTOR,
        QueueOptions::GPU_SELECTOR };

    for (auto device_selector :  QueueOptionsDevices) {
        dpnp_queue_initialize_c(device_selector);
        double* result1 = (double*)dpnp_memory_alloc_c(size * sizeof(double));
        double* result2 = (double*)dpnp_memory_alloc_c(size * sizeof(double));

        dpnp_srand_c(seed);
        dpnp_rng_uniform_c<double>(result1, low, high, size);

        dpnp_srand_c(seed);
        dpnp_rng_uniform_c<double>(result2, low, high, size);

        for (size_t i = 0; i < size; ++i)
        {
            EXPECT_NEAR (result1[i], result2[i], 0.004);
        }
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
