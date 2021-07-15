#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

const int reprtitions = 15;

sycl::event divide_array_by_array(sycl::queue q,
                                      std::vector<sycl::event> &deps,
                                      const size_t size,
                                      double* a,
                                      double* b,
                                      double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] / b[i];}); 
    });

    return event;  
}

sycl::event divide_array_scalar(sycl::queue q,
                                    std::vector<sycl::event> &deps,
                                    const size_t size,
                                    double a,
                                    double* b,
                                    double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a / b[i];});
    });

    return event;  
}

sycl::event log_sycl(sycl::queue q,
                         std::vector<sycl::event> &deps,
                         const size_t size,
                         double* a,
                         double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = sycl::log(a[i]);});
    });

    return event;
}

sycl::event multiply_array_by_scalar(sycl::queue q,
                                         std::vector<sycl::event> &deps,
                                         const size_t size,
                                         double* a,
                                         double b,
                                         double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] * b;});
    });

    return event;
}

sycl::event multiply_arrays(sycl::queue q,
                                std::vector<sycl::event> &deps,
                                const size_t size,
                                double* a,
                                double* b,
                                double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] * b[i];});
    });

    return event;
}

sycl::event sqrt(sycl::queue q,
                     std::vector<sycl::event> &deps,
                     const size_t size,
                     double* a,
                     double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = sycl::sqrt(a[i]);});
    });

    return event;
}

sycl::event subtract(sycl::queue q,
                            std::vector<sycl::event> &deps,
                            const size_t size,
                            double* a,
                            double* b,
                            double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] - b[i];});
    });

    return event;
}

sycl::event add(sycl::queue q,
                    std::vector<sycl::event> &deps,
                    const size_t size,
                    double* a,
                    double* b,
                    double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] + b[i];});
    });

    return event;
}

sycl::event erf(sycl::queue q,
                         std::vector<sycl::event> &deps,
                         const size_t size,
                         double* a,
                         double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = sycl::erf(a[i]);});
    });

    return event;
}

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   const double rate,
                   const double vol,
                   const size_t size,
                   sycl::queue q,
                   bool sync = false)
{
    double mr = -rate;
    double vol_vol_two = vol * vol * 2;

    // ------------ p_div_s = price / strike ----------
    std::vector<sycl::event> depsvents_p_div_s;
    double* p_div_s = sycl::malloc_shared<double>(size, q);
    sycl::event e_div = divide_array_by_array(q, depsvents_p_div_s, size, price, strike, p_div_s);

    if (sync)
        e_div.wait();

    // ------------ a = log(p_div_s) ------------
    std::vector<sycl::event> depsvents_a;
    if (!sync) 
        depsvents_a.push_back(e_div);
    double* a = sycl::malloc_shared<double>(size, q);
    sycl::event e_a = log_sycl(q, depsvents_a, size, p_div_s, a);

    if (sync)
        e_a.wait();

    // ------------ b = t * mr ------------
    std::vector<sycl::event> depsvents_b;
    double* b = sycl::malloc_shared<double>(size, q);
    sycl::event e_b = multiply_array_by_scalar(q, depsvents_b, size, t, mr, b);

    if (sync)
        e_b.wait();

    //------------ z = t * vol_vol_two ------------
    std::vector<sycl::event> depsvents_z;
    double* z = sycl::malloc_shared<double>(size, q);
    sycl::event e_z = multiply_array_by_scalar(q, depsvents_z, size, t, vol_vol_two, z);
    if (sync)
        e_z.wait();

    //------------ c = quartnes * z ------------
    std::vector<sycl::event> depsvents_c;
    if (!sync)
        depsvents_c.push_back(e_z);
    double* c = sycl::malloc_shared<double>(size, q);
    sycl::event e_c = multiply_array_by_scalar(q, depsvents_c, size, z, 0.25, c);
    if (sync)
        e_c.wait();

    //------------ sqrt(z) ------------
    std::vector<sycl::event> depsvents_sqrt_z;
    if (!sync)
        depsvents_sqrt_z.push_back(e_z);
    double* sqrt_z = sycl::malloc_shared<double>(size, q);
    sycl::event e_sqrt_z = sqrt(q, depsvents_sqrt_z, size, z, sqrt_z);
    if (sync)
        e_sqrt_z.wait();

    //------------ y = ones / sqrt(z) ------------
    std::vector<sycl::event> depsvents_y;
    if (!sync)
        depsvents_y.push_back(e_sqrt_z);
    double* y = sycl::malloc_shared<double>(size, q);
    sycl::event e_y = divide_array_scalar(q, depsvents_y, size, 1, sqrt_z, y);
    if (sync)
        e_y.wait();

    //------------ a_sub_b = a - b ------------
    std::vector<sycl::event> depsvents_a_sub_b;
    if (!sync) 
    {
        depsvents_a_sub_b.insert(depsvents_a_sub_b.end(), {e_a, e_b});
    }
    double* a_sub_b = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b = subtract(q, depsvents_a_sub_b, size, a, b, a_sub_b);
    if (sync)
        e_a_sub_b.wait();

    //------------ a_sub_b_add_c = a_sub_b + c ------------
    std::vector<sycl::event> depsvents_a_sub_b_add_c;
    if (!sync) 
    {
        depsvents_a_sub_b_add_c.insert(depsvents_a_sub_b_add_c.end(), {e_c, e_a_sub_b});
    }
    double* a_sub_b_add_c = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b_add_c = add(q, depsvents_a_sub_b_add_c, size, a_sub_b, c, a_sub_b_add_c);
    if (sync)
        e_a_sub_b_add_c.wait();

    //------------ w1 = a_sub_b_add_c * y ------------
    std::vector<sycl::event> depsvents_w1;
    if (!sync) {
        depsvents_a_sub_b_add_c.insert(depsvents_a_sub_b_add_c.end(), {e_a_sub_b_add_c, e_y});
    }
    double* w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_w1 = multiply_arrays(q, depsvents_w1, size, a_sub_b, c, a_sub_b_add_c);
    if (sync)
        e_w1.wait();

     //------------ a_sub_b_sub_c = a_sub_b - c ------------
    std::vector<sycl::event> depsvents_a_sub_b_sub_c;
    if (!sync) 
    {
        depsvents_a_sub_b_add_c.insert(depsvents_a_sub_b_add_c.end(), {e_c, e_a_sub_b});
    }
    double* a_sub_b_sub_c = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b_sub_c = subtract(q, depsvents_a_sub_b_sub_c, size, a_sub_b, c, a_sub_b_sub_c);
    if (sync)
        e_a_sub_b_sub_c.wait();

    //------------ w2 = a_sub_b_sub_c * y ------------
    std::vector<sycl::event> depsvents_w2;
    if (!sync) {
        depsvents_a_sub_b_add_c.insert(depsvents_a_sub_b_add_c.end(), {e_a_sub_b_sub_c, e_y});
    }
    double* w2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_w2 = multiply_arrays(q, depsvents_w2, size, a_sub_b, c, a_sub_b_sub_c);
    if (sync)
        e_w2.wait();

    //------------ erf_w1 = erf(w1) ------------
    std::vector<sycl::event> depsvents_erf_w1;
    if (!sync) {
        depsvents_a_sub_b_add_c.push_back(e_w1);
    }
    double* erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_erf_w1 = erf(q, depsvents_erf_w1, size, w1, erf_w1);
    if (sync)
        e_erf_w1.wait();

    //------------ halfs_mul_erf_w1 = halfs * erf_w1 ------------
    std::vector<sycl::event> depsvents_halfs_mul_erf_w1;
    if (!sync) {
        depsvents_a_sub_b_add_c.push_back(e_erf_w1);
    }
    double* halfs_mul_erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_halfs_mul_erf_w1 = multiply_array_by_scalar(q, depsvents_halfs_mul_erf_w1, size, erf_w1, 0.5, halfs_mul_erf_w1);
    if (sync)
        e_halfs_mul_erf_w1.wait();
    
    sycl::free(p_div_s, q);
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(z, q);
    sycl::free(c, q);
    sycl::free(sqrt_z, q);
    sycl::free(y, q);
    sycl::free(a_sub_b, q);

    // free memory
}

double random(int a, int b) 
{
    return (double) rand() / (double) RAND_MAX * (b - a) + b;
}

double median(std::vector<double> times)
{
    std::sort(times.begin(), times.end());
    if (times.size() % 2)
        return times[times.size()/2];
    return times[times.size()/2 - 1] + times[times.size()/2] / 2;
}

int main() {

    const size_t SIZE = 20;
    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;
    
    sycl::queue q{sycl::gpu_selector{}};

    double *price = sycl::malloc_shared<double>(SIZE, q);
    double *strike = sycl::malloc_shared<double>(SIZE, q);
    double *t = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = random(PL, PH); 
        strike[i] = random(SL, SH);
        t[i] =  random(TL, TH);
    }
    
    //------------ performanse mesure ------------
    std::vector<double> async_times;
    for (int i = 0; i < reprtitions; ++i) // reprtitions
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, SIZE, q);

        auto t2 = std::chrono::high_resolution_clock::now();
        async_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count());
    }

    std::vector<double> sync_times;
    for (int i = 0; i < reprtitions; ++i) 
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, SIZE, q, true);

        auto t2 = std::chrono::high_resolution_clock::now();
        sync_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count());
    }

    std::cout << "Async time: " << median(async_times) << " ms.\n";
    std::cout << "Sync time: " << median(sync_times) << " ms.\n";

    sycl::free(price, q);
    sycl::free(strike, q);
    sycl::free(t, q);
}