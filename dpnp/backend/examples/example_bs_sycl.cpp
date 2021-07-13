#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

cl::sycl::event divide_sycl(cl::sycl::queue q,
                            const size_t size,
                            double* price,
                            double* strike,
                            double* p_div_s)
{
    double* P = price;
    double* S = strike;

    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { p_div_s[i] = P[i] / S[i];});
    });

    return event;  
}

cl::sycl::event log_sycl(cl::sycl::queue q,
                         cl::sycl::event e,
                         const size_t size,
                         double* p_div_s,
                         double* a)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { a[i] = cl::sycl::log(p_div_s[i]);});
    });

    return event;
}


cl::sycl::event multiply_sycl(cl::sycl::queue q,
                              const size_t size,
                              double* t,
                              double* mr,
                              double* b)
{
    double* T = t;

    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { b[i] = T[i] * mr[0];});
    });

    return event;
}

                            

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   const double rate,
                   const double vol,
                   const size_t size,
                   bool is_sync,
                   cl::sycl::queue q)
{
    if (is_sync) {
        double* mr = cl::sycl::malloc_shared<double>(1, q);
        mr[0] = -rate;

        double* vol_vol_two = cl::sycl::malloc_shared<double>(1, q);
        vol_vol_two[0] = vol * vol * 2;

        double* quarter = cl::sycl::malloc_shared<double>(1, q);
        quarter[0] = 0.25;

        double* one = cl::sycl::malloc_shared<double>(1, q);
        one[0] = 1.;

        double* half = cl::sycl::malloc_shared<double>(1, q);
        half[0] = 0.5;

        cl::sycl::event event; // !!!

        // ------------ dividing ----------
        double* p_div_s = sycl::malloc_shared<double>(size, q);
        event = divide_sycl(q, size, price, strike, p_div_s);

        // -------------- log ------------
        double* a = sycl::malloc_shared<double>(size, q);
        event = log_sycl(q, event, size, p_div_s, a);

        // ------------ multiply ----------
        double* b = cl::sycl::malloc_shared<double>(size, q);
        cl::sycl::event e_b = multiply_sycl(q, size, t, mr, b);
        e_b.wait();

        for (int i = 0; i < size; ++i)
            std::cout << b[i] << "\n";

        
        cl::sycl::free(p_div_s, q);
        cl::sycl::free(a, q);
        cl::sycl::free(b, q);
    }
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
    bool is_sync = true;

    //------ example -------
    double *price = sycl::malloc_shared<double>(SIZE, q);
    double *strike = sycl::malloc_shared<double>(SIZE, q);
    double *t = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = rand() % (PH - PL) + PL;
        strike[i] = rand() % (SH - SL) + SL;
        t[i] = rand() % (TH -TL) + TL;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    black_scholes(price, strike, t, RISK_FREE, VOLATILITY, SIZE, is_sync, q);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    std::cout << duration;

    cl::sycl::free(price, q);
    cl::sycl::free(strike, q);
    cl::sycl::free(t, q);
}