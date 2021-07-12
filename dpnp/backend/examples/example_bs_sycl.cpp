#include <CL/sycl.hpp>
#include <iostream>

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
                         const size_t size,
                         double* p_div_s,
                         double* a)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { a[i] = cl::sycl::log(p_div_s[i]);});
    });

    return event;
}


void black_scholes(double* price,
                   double* strike,
                   const double rate,
                   const double vol,
                   const size_t size,
                   cl::sycl::queue q)
{
    // ------------ dividing ----------
    double* p_div_s = sycl::malloc_shared<double>(size, q);

    cl::sycl::event event;
    event = divide_sycl(q, size, price, strike, p_div_s);
    event.wait();

    for (int i = 0; i < size; ++i)
        std::cout << price[i] << " / " << strike[i] << " = " << p_div_s[i] << "\n";

    // ------------ log ----------
    double* a = sycl::malloc_shared<double>(size, q);

    event = log_sycl(q, size, p_div_s, a);
    event.wait();

    for (int i = 0; i < size; ++i)
        std::cout << a[i] << "\n";
    
    cl::sycl::free(p_div_s, q);
    cl::sycl::free(a, q);
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

    //------ example -------
    double *price = sycl::malloc_shared<double>(SIZE, q);
    double *strike = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = rand() % (PH - PL) + PL;
        strike[i] = rand() % (SH - SL) + SL;
    }

    black_scholes(price, strike, RISK_FREE, VOLATILITY, SIZE, q);

    cl::sycl::free(price, q);
    cl::sycl::free(strike, q);
}
