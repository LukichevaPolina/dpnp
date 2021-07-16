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

sycl::event add_arrays(sycl::queue q,
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

sycl::event add_scalar_to_array(sycl::queue q,
                    std::vector<sycl::event> &deps,
                    const size_t size,
                    double* a,
                    double b,
                    double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] + b;});
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

sycl::event exp(sycl::queue q,
                std::vector<sycl::event> &deps,
                const size_t size,
                double* a,
                double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);
	    cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = sycl::exp(a[i]);});
    });

    return event;
}

void black_scholes(double* price,
                   double* strike,
                   double* t,
                   const double rate,
                   const double vol,
                //    double* call,
                //    double* put,
                   const size_t size,
                   sycl::queue q,
                   bool sync = false)
{
    double mr = -rate;
    double vol_vol_two = vol * vol * 2;

    // ------------ p_div_s = price / strike ----------
    std::vector<sycl::event> dep_events_p_div_s;
    double* p_div_s = sycl::malloc_shared<double>(size, q);
    sycl::event e_div = divide_array_by_array(q, dep_events_p_div_s, size, price, strike, p_div_s);

    if (sync)
        e_div.wait();

    // ------------ a = log(p_div_s) ------------
    std::vector<sycl::event> dep_events_a;
    if (!sync) 
        dep_events_a.push_back(e_div);
    double* a = sycl::malloc_shared<double>(size, q);
    sycl::event e_a = log_sycl(q, dep_events_a, size, p_div_s, a);

    if (sync)
        e_a.wait();
    
    sycl::free(p_div_s, q);

    // ------------ b = t * mr ------------
    std::vector<sycl::event> dep_events_b;
    double* b = sycl::malloc_shared<double>(size, q);
    sycl::event e_b = multiply_array_by_scalar(q, dep_events_b, size, t, mr, b);

    if (sync)
        e_b.wait();

    //------------ z = t * vol_vol_two ------------
    std::vector<sycl::event> dep_events_z;
    double* z = sycl::malloc_shared<double>(size, q);
    sycl::event e_z = multiply_array_by_scalar(q, dep_events_z, size, t, vol_vol_two, z);
    if (sync)
        e_z.wait();

    //------------ c = quartnes * z ------------
    std::vector<sycl::event> dep_events_c;
    if (!sync)
        dep_events_c.push_back(e_z);
    double* c = sycl::malloc_shared<double>(size, q);
    sycl::event e_c = multiply_array_by_scalar(q, dep_events_c, size, z, 0.25, c);
    if (sync)
        e_c.wait();

    //------------ sqrt(z) ------------
    std::vector<sycl::event> dep_events_sqrt_z;
    if (!sync)
        dep_events_sqrt_z.push_back(e_z);
    double* sqrt_z = sycl::malloc_shared<double>(size, q);
    sycl::event e_sqrt_z = sqrt(q, dep_events_sqrt_z, size, z, sqrt_z);
    if (sync)
        e_sqrt_z.wait();
    
    sycl::free(z, q);

    //------------ y = ones / sqrt(z) ------------
    std::vector<sycl::event> dep_events_y;
    if (!sync)
        dep_events_y.push_back(e_sqrt_z);
    double* y = sycl::malloc_shared<double>(size, q);
    sycl::event e_y = divide_array_scalar(q, dep_events_y, size, 1, sqrt_z, y);
    if (sync)
        e_y.wait();

    sycl::free(sqrt_z, q);

    //------------ a_sub_b = a - b ------------
    std::vector<sycl::event> dep_events_a_sub_b;
    if (!sync) 
    {
        dep_events_a_sub_b.insert(dep_events_a_sub_b.end(), {e_a, e_b});
    }
    double* a_sub_b = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b = subtract(q, dep_events_a_sub_b, size, a, b, a_sub_b);
    if (sync)
        e_a_sub_b.wait();
    
    sycl::free(a, q);

    //------------ a_sub_b_add_c = a_sub_b + c ------------
    std::vector<sycl::event> dep_events_a_sub_b_add_c;
    if (!sync) 
    {
        dep_events_a_sub_b_add_c.insert(dep_events_a_sub_b_add_c.end(), {e_c, e_a_sub_b});
    }
    double* a_sub_b_add_c = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b_add_c = add_arrays(q, dep_events_a_sub_b_add_c, size, a_sub_b, c, a_sub_b_add_c);
    if (sync)
        e_a_sub_b_add_c.wait();

    //------------ w1 = a_sub_b_add_c * y ------------
    std::vector<sycl::event> dep_events_w1;
    if (!sync) {
        dep_events_a_sub_b_add_c.insert(dep_events_a_sub_b_add_c.end(), {e_a_sub_b_add_c, e_y});
    }
    double* w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_w1 = multiply_arrays(q, dep_events_w1, size, a_sub_b, c, a_sub_b_add_c);
    if (sync)
        e_w1.wait();
    
    sycl::free(a_sub_b_add_c, q);

     //------------ a_sub_b_sub_c = a_sub_b - c ------------
    std::vector<sycl::event> dep_events_a_sub_b_sub_c;
    if (!sync) 
    {
        dep_events_a_sub_b_add_c.insert(dep_events_a_sub_b_add_c.end(), {e_c, e_a_sub_b});
    }
    double* a_sub_b_sub_c = sycl::malloc_shared<double>(size, q);
    sycl::event e_a_sub_b_sub_c = subtract(q, dep_events_a_sub_b_sub_c, size, a_sub_b, c, a_sub_b_sub_c);
    if (sync)
        e_a_sub_b_sub_c.wait();

    sycl::free(a_sub_b, q);
    sycl::free(c, q);

    //------------ w2 = a_sub_b_sub_c * y ------------
    std::vector<sycl::event> dep_events_w2;
    if (!sync) {
        dep_events_a_sub_b_add_c.insert(dep_events_a_sub_b_add_c.end(), {e_a_sub_b_sub_c, e_y});
    }
    double* w2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_w2 = multiply_arrays(q, dep_events_w2, size, a_sub_b, c, a_sub_b_sub_c);
    if (sync)
        e_w2.wait();
    
    sycl::free(a_sub_b_sub_c, q);
    sycl::free(y, q);

    //------------ erf_w1 = erf(w1) ------------
    std::vector<sycl::event> dep_events_erf_w1;
    if (!sync) {
        dep_events_erf_w1.push_back(e_w1);
    }
    double* erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_erf_w1 = erf(q, dep_events_erf_w1, size, w1, erf_w1);
    if (sync)
        e_erf_w1.wait();
    
    sycl::free(w1, q);

    //------------ halfs_mul_erf_w1 = halfs * erf_w1 ------------
    std::vector<sycl::event> dep_events_halfs_mul_erf_w1;
    if (!sync) {
        dep_events_halfs_mul_erf_w1.push_back(e_erf_w1);
    }
    double* halfs_mul_erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_halfs_mul_erf_w1 = multiply_array_by_scalar(q, dep_events_halfs_mul_erf_w1, size, erf_w1, 0.5, halfs_mul_erf_w1);
    if (sync)
        e_halfs_mul_erf_w1.wait();

    sycl::free(erf_w1, q);

    //------------ d1 = half + halfs_mul_erf_w1 ------------
    std::vector<sycl::event> dep_events_d1;
    if (!sync) {
        dep_events_a_sub_b_add_c.push_back(e_halfs_mul_erf_w1);
    }
    double* d1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_d1 = add_scalar_to_array(q, dep_events_d1, size, halfs_mul_erf_w1, 0.5, d1);
    if (sync)
        e_d1.wait();
    
    sycl::free(halfs_mul_erf_w1, q);

    //------------ erf_w2 = erf(w2) ------------
    std::vector<sycl::event> dep_events_erf_w2;
    if (!sync) {
        dep_events_erf_w2.push_back(e_w2);
    }
    double* erf_w2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_erf_w2 = erf(q, dep_events_erf_w2, size, w2, erf_w2);
    if (sync)
        e_erf_w2.wait();

    sycl::free(w2, q);

    //------------ halfs_mul_erf_w2 = half * erf_w2 ------------
    std::vector<sycl::event> dep_events_halfs_mul_erf_w2;
    if (!sync) {
        dep_events_halfs_mul_erf_w2.push_back(e_erf_w2);
    }
    double* halfs_mul_erf_w2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_halfs_mul_erf_w2 = multiply_array_by_scalar(q, dep_events_halfs_mul_erf_w2, size, erf_w2, 0.5, halfs_mul_erf_w2);
    if (sync)
        e_halfs_mul_erf_w2.wait();
    
    sycl::free(erf_w2, q);

    //------------ d2 = halfs + halfs_mul_erf_w2 ------------
    std::vector<sycl::event> dep_events_d2;
    if (!sync) {
        dep_events_d2.push_back(e_halfs_mul_erf_w2);
    }
    double* d2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_d2 = add_scalar_to_array(q, dep_events_d2, size, halfs_mul_erf_w2, 0.5, d2);
    if (sync)
        e_d2.wait();
    
    sycl::free(halfs_mul_erf_w2, q);

    //------------ exp_b = exp(b) ------------
    std::vector<sycl::event> dep_events_exp_b;
    if (!sync) {
        dep_events_exp_b.push_back(e_b);
    }
    double* exp_b = sycl::malloc_shared<double>(size, q);
    sycl::event e_exp_b = exp(q, dep_events_exp_b, size, b, exp_b);
    if (sync)
        e_exp_b.wait();
    
    sycl::free(b, q);

    //------------ Se = exp_b * strike ------------
    std::vector<sycl::event> dep_events_Se;
    if (!sync) {
        dep_events_Se.push_back(e_exp_b);
    }
    double* Se = sycl::malloc_shared<double>(size, q);
    sycl::event e_Se = multiply_arrays(q, dep_events_Se, size, exp_b, strike, Se);
    if (sync)
        e_Se.wait();
    
    sycl::free(exp_b, q);
    
    //------------ P_mul_d1 = price * d1 ------------
    std::vector<sycl::event> dep_events_P_mul_d1;
    if (!sync) {
        dep_events_P_mul_d1.push_back(e_d1);
    }
    double* P_mul_d1 = sycl::malloc_shared<double>(size, q);
    sycl::event e_P_mul_d1 = multiply_arrays(q, dep_events_P_mul_d1, size, price, d1, P_mul_d1);
    if (sync)
        e_P_mul_d1.wait();
    
    sycl::free(d1, q);

    //------------ Se_mul_d2 = Se * d2 ------------
    std::vector<sycl::event> dep_events_Se_mul_d2;
    if (!sync) {
        dep_events_Se_mul_d2.insert(dep_events_Se_mul_d2.end(), {e_Se, e_d2});
    }
    double* Se_mul_d2 = sycl::malloc_shared<double>(size, q);
    sycl::event e_Se_mul_d2 = multiply_arrays(q, dep_events_Se_mul_d2, size, Se, d2, Se_mul_d2);
    if (sync)
        e_Se_mul_d2.wait();
    
    sycl::free(d2, q);

    //------------ r = P_mul_d1 - Se_mul_d2 ------------
    std::vector<sycl::event> dep_events_r;
    if (!sync) {
        dep_events_r.insert(dep_events_r.end(), {e_P_mul_d1, e_Se_mul_d2});
    }
    double* r = sycl::malloc_shared<double>(size, q);
    sycl::event e_r = subtract(q, dep_events_r, size, P_mul_d1, Se_mul_d2, r);
    if (sync)
        e_r.wait();

    sycl::free(P_mul_d1, q);
    sycl::free(Se_mul_d2, q);
    
    //dpnp_copyto_c<double, double>(call, r, size); // call[:] = r

    //------------ r_sub_P = r - P ------------
    std::vector<sycl::event> dep_events_r_sub_P;
    if (!sync) {
        dep_events_r_sub_P.push_back(e_r);
    }
    double* r_sub_P = sycl::malloc_shared<double>(size, q);
    sycl::event e_r_sub_P = subtract(q, dep_events_r_sub_P, size, r, price, r_sub_P);
    if (sync)
        e_r_sub_P.wait();

    //sycl::free(r, q);

    //------------ r_sub_P_add_Se = r_sub_P + Se ------------
    std::vector<sycl::event> dep_events_r_sub_P_add_Se;
    if (!sync) {
        dep_events_r_sub_P_add_Se.insert(dep_events_r_sub_P_add_Se.end(), {e_r_sub_P, e_Se});
    }
    double* r_sub_P_add_Se = sycl::malloc_shared<double>(size, q);
    sycl::event e_r_sub_P_add_Se = add_arrays(q, dep_events_r_sub_P_add_Se, size, r_sub_P, Se, r_sub_P_add_Se);
    if (sync)
        e_r_sub_P_add_Se.wait();
    
    sycl::free(r_sub_P, q);
    sycl::free(Se, q);

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
