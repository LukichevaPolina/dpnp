#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

cl::sycl::event divide_sycl(cl::sycl::queue q,
                            std::vector<cl::sycl::event> &dep_e,
                            const size_t size,
                            double* a,
                            double* b,
                            double* y)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep_e);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] / b[i];});
    });

    return event;  
}

cl::sycl::event log_sycl(cl::sycl::queue q,
                         std::vector<cl::sycl::event> &dep_e,
                         const size_t size,
                         double* a,
                         double* y)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep_e);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = cl::sycl::log(a[i]);});
    });

    return event;
}

cl::sycl::event multiply_array_by_scalar(cl::sycl::queue q,
                                         std::vector<cl::sycl::event> &dep_e,
                                         const size_t size,
                                         double* a,
                                         double b,
                                         double* y)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep_e);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] * b;});
    });

    return event;
}

cl::sycl::event sqrt(cl::sycl::queue q,
                     std::vector<cl::sycl::event> &dep_e,
                     const size_t size,
                     double* a,
                     double* y)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep_e);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = cl::sycl::sqrt(a[i]);});
    });

    return event;
}

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   const double rate,
                   const double vol,
                   const size_t size,
                   cl::sycl::queue q,
                   bool is_sync = false)
{
    double mr = - rate;

    double vol_vol_two = vol * vol * 2;

    double quarter = 0.25;

    double one = 1.;

    double half = 0.5;

    // ------------ p_div_s = price / strike ----------
    std::vector<cl::sycl::event> dep_events_p_div_s;
    double* p_div_s = sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_div = divide_sycl(q, dep_events_p_div_s, size, price, strike, p_div_s);

    if (is_sync)
        e_div.wait();

    // ------------ a = log(p_div_s) ------------
    std::vector<cl::sycl::event> dep_events_a;
    if (!is_sync) 
        dep_events_a.push_back(e_div);
    double* a = sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_log = log_sycl(q, dep_events_a, size, p_div_s, a);

    if (is_sync)
        e_log.wait();

    // ------------ b = t * mr ------------
    std::vector<cl::sycl::event> dep_events_b;
    double* b = cl::sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_b = multiply_array_by_scalar(q, dep_events_b, size, t, mr, b);

    if (is_sync)
        e_b.wait();

    //------------ z = t * vol_vol_two ------------
    std::vector<cl::sycl::event> dep_events_z;
    double* z = cl::sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_z = multiply_array_by_scalar(q, dep_events_z, size, t, vol_vol_two, z);
    if (is_sync)
        e_z.wait();

    //------------ c = quartnes * z ------------
    std::vector<cl::sycl::event> dep_events_c;
    if (!is_sync)
        dep_events_c.push_back(e_z);
    double* c = cl::sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_c = multiply_array_by_scalar(q, dep_events_c, size, z, quarter, c);
    if (is_sync)
        e_c.wait();

    //------------ sqrt(z) ------------
    std::vector<cl::sycl::event> dep_events_sqrt_z;
    if (!is_sync)
        dep_events_sqrt_z.push_back(e_z);
    double* sqrt_z = cl::sycl::malloc_shared<double>(size, q);
    cl::sycl::event e_sqrt_z = sqrt(q, dep_events_sqrt_z, size, z, sqrt_z);
    if (is_sync)
        e_sqrt_z.wait();

    //------------ y = ones / sqrt(z) ------------
    // std::vector<cl::sycl::event> dep_events_y;
    // if (!is_sync)
    //     dep_events_sqrt_z.push_back(e_sqrt_z);
    // double* y = cl::sycl::malloc_shared<double>(size, q);
    // cl::sycl::event e_y = divide_sycl(q, dep_events_y, size, &one, sqrt_z, y);
    // if (is_sync)
    //     e_y.wait();
    
    cl::sycl::free(p_div_s, q);
    cl::sycl::free(a, q);
    cl::sycl::free(b, q);
    cl::sycl::free(z, q);
    cl::sycl::free(c, q);
    cl::sycl::free(sqrt_z, q);
    //cl::sycl::free(y, q);
}

int median(std::vector<int> performance)
{
    std::sort(performance.begin(), performance.end());
    return performance[performance.size()/2];
}

int main() {

    const size_t SIZE = 20;
    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;
    
    cl::sycl::queue q{sycl::gpu_selector{}};

    double *price = sycl::malloc_shared<double>(SIZE, q);
    double *strike = sycl::malloc_shared<double>(SIZE, q);
    double *t = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = (double) rand() / (double) RAND_MAX * (PH - PL) + PL;
        strike[i] = (double) rand() / (double) RAND_MAX * (SH - SL) + SL;
        t[i] = (double) rand() / (double) RAND_MAX * (TH -TL) + TL;
    }
    
    //------------ performanse mesure ------------
    std::vector<int> async_performances;
    for (int i = 0; i < 15; ++i) 
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, SIZE, q);

        auto t2 = std::chrono::high_resolution_clock::now();
        async_performances.push_back(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count());
    }

    std::vector<int> sync_performances;
    for (int i = 0; i < 15; ++i) 
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, SIZE, q, true);

        auto t2 = std::chrono::high_resolution_clock::now();
        sync_performances.push_back(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count());
    }

    std::cout << "Async performance: " << median(async_performances) << "\n";
    std::cout << "Sync performance: " << median(sync_performances) << "\n";

    cl::sycl::free(price, q);
    cl::sycl::free(strike, q);
    cl::sycl::free(t, q);
}