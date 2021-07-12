#include <CL/sycl.hpp>
#include <iostream>

double* divide_sycl(double* price,
                    double* strike,
                    const size_t size,
                    cl::sycl::queue q)
{
    double* P = sycl::malloc_shared<double>(size, q);
    double* S = sycl::malloc_shared<double>(size, q);
    double* p_div_s = sycl::malloc_shared<double>(size, q);

    P = price;
    S = strike;

    q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { p_div_s[i] = P[i] / S[i];});
    });
    q.wait();
    
    return p_div_s;
    
}

double* log_sycl(const size_t size,
                 double* p_div_s,
                 cl::sycl::queue q)
{
    double* log_ = sycl::malloc_shared<double>(size, q);
    q.submit([&](sycl::handler &cgh) {
	    cgh.parallel_for(size, [=](sycl::item<1> i) { log_[i] = cl::sycl::log(p_div_s[i]);});
    });
    q.wait();

    return log_;
}


void black_scholes(double* price,
                   double* strike,
                   const double rate,
                   const double vol,
                   const size_t size)
{
    cl::sycl::queue q{sycl::gpu_selector{}};

    // ------------ dividing ----------
    double(*p_div_s) = new double[size];
    p_div_s = divide_sycl(price, strike, size, q);

    for (int i = 0; i < size; ++i)
        std::cout << price[i] << " / " << strike[i] << " = " << p_div_s[i] << "\n";

    // ------------ log ----------
    double(*log_) = new double[size];
    log_ = log_sycl(size, p_div_s, q);

    for (int i = 0; i < size; ++i)
        std::cout << log_[i] << "\n";
}


int main() {

    const size_t SIZE = 20;
    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;
    
    //------ example -------
    double(*price) = new double[SIZE];
    double(*strike) = new double[SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = rand() % (PL + PH) + PL;
        strike[i] = rand() % (SL + SH) + SL;
    }

    black_scholes(price, strike, RISK_FREE, VOLATILITY, SIZE);
}