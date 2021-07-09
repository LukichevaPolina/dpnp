#include <CL/sycl.hpp>
#include <iostream>

void black_scholes(double* price,
                   double* strike,
                   const double rate,
                   const double vol,
                   const size_t size)
{
    // ------------ dividing ----------
    double* pr_div_st = new double[size];
    
    cl::sycl::queue q{sycl::gpu_selector{}};

    cl::sycl::buffer P_buf(reinterpret_cast<double*> (price), cl::sycl::range(size)); 
    cl::sycl::buffer S_buf(reinterpret_cast<double*> (strike), cl::sycl::range(size)); 
    cl::sycl::buffer p_div_s_buf(reinterpret_cast<double*> (pr_div_st), cl::sycl::range(size)); 

    q.submit([&](sycl::handler &cgh) {
        auto P = P_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto S = S_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto p_div_s = p_div_s_buf.get_access<cl::sycl::access::mode::write>(cgh);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { p_div_s[i] = P[i] / S[i];});
    });
    q.wait();

    auto p_div_s = p_div_s_buf.get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < size; ++i)
        std::cout << price[i] << " / " << strike[i] << " = " << p_div_s[i] << "\n";


    // ------------ log ----------
    double* log = new double[size];

    cl::sycl::buffer log_buf(reinterpret_cast<double*> (log), cl::sycl::range(size));
    q.submit([&](sycl::handler &cgh) {
        auto p_div_s = p_div_s_buf.get_access<cl::sycl::access::mode::read>(cgh);
        auto log_ = log_buf.get_access<cl::sycl::access::mode::write>(cgh);   
	    cgh.parallel_for(size, [=](sycl::item<1> i) { log_[i] = cl::sycl::log(p_div_s[i]);});
    });
    q.wait();
    auto log_ = log_buf.get_access<cl::sycl::access::mode::read>();
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
    
    //------ example-------
    double(*price) = new double[SIZE];
    double(*strike) = new double[SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = rand() % (PL + PH) + PL;
        strike[i] = rand() % (SL + SH) + SL;
    }

    black_scholes(price, strike, RISK_FREE, VOLATILITY, SIZE);
}