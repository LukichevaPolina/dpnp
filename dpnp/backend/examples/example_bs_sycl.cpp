#include <CL/sycl.hpp>
#include <iostream>

cl::sycl::event divide_sycl(cl::sycl::queue q,
                            const size_t size,
                            double* price,
                            double* strike,
                            double* p_div_s)
{
    double* P = sycl::malloc_shared<double>(size, q);
    double* S = sycl::malloc_shared<double>(size, q);

    P = price;
    S = strike;

    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { p_div_s[i] = P[i] / S[i];});
    });

    cl::sycl::free(P, q);
    cl::sycl::free(S, q);

    return event;  
}

cl::sycl::event log_sycl(const size_t size,
                         double* p_div_s,
                         double* log_,
                         cl::sycl::queue q)
{
    cl::sycl::event event = q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { log_[i] = cl::sycl::log(p_div_s[i]);});
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
    double(*p_div_s) = sycl::malloc_shared<double>(size, q);

    cl::sycl::event event;
    event = divide_sycl(price, strike, size, p_div_s, q);
    event.wait();

    for (int i = 0; i < size; ++i)
        std::cout << price[i] << " / " << strike[i] << " = " << p_div_s[i] << "\n";

    // ------------ log ----------
    double(*log_) = sycl::malloc_shared<double>(size, q);

    event = log_sycl(size, p_div_s, log_, q);
    event.wait();

    for (int i = 0; i < size; ++i)
        std::cout << log_[i] << "\n";
    
    cl::sycl::free(p_div_s, q);
    cl::sycl::free(log_, q);
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
    double(*price) = sycl::malloc_shared<double>(SIZE, q);
    double(*strike) = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = rand() % (PH - PL) + PL;
        strike[i] = rand() % (SH - SL) + SL;
    }

    black_scholes(price, strike, RISK_FREE, VOLATILITY, SIZE, q);

    cl::sycl::free(price, q);
    
    cl::sycl::free(strike, q);
}
