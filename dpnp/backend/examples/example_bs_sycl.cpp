#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

const int reprtitions = 1;

sycl::event divide(sycl::queue q,
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

sycl::event divide_array_by_scalar(sycl::queue q,
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

sycl::event log(sycl::queue q,
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

sycl::event multiply(sycl::queue q,
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
<<<<<<< HEAD
        cgh.depends_on(deps);
=======
	    cgh.depends_on(deps);
>>>>>>> b95dea7f99839d0ad7e0d4afeee422907bc5a810
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
<<<<<<< HEAD
        cgh.depends_on(deps);
=======
	    cgh.depends_on(deps);
>>>>>>> b95dea7f99839d0ad7e0d4afeee422907bc5a810
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
                   double* call,
                   double* put,
                   const size_t size,
                   sycl::queue q,
                   bool sync = false)
{
    double mr = -rate;
    double vol_vol_two = vol * vol * 2;

    // ------------ p_div_s = price / strike ----------
    std::vector<sycl::event> p_div_s_deps;
    double* p_div_s = sycl::malloc_shared<double>(size, q);
    sycl::event p_div_s_event = divide(q, p_div_s_deps, size, price, strike, p_div_s);

    if (sync)
        p_div_s_event.wait();

    // ------------ a = log(p_div_s) ------------
    std::vector<sycl::event> a_deps;
    if (!sync) 
        a_deps.push_back(p_div_s_event);
    double* a = sycl::malloc_shared<double>(size, q);
    sycl::event a_event = log(q, a_deps, size, p_div_s, a);

    if (sync)
        a_event.wait();

    sycl::free(p_div_s, q);

    // ------------ b = t * mr ------------
    std::vector<sycl::event> b_deps;
    double* b = sycl::malloc_shared<double>(size, q);
    sycl::event b_event = multiply_array_by_scalar(q, b_deps, size, t, mr, b);

    if (sync)
        b_event.wait();

    //------------ z = t * vol_vol_two ------------
    std::vector<sycl::event> z_deps;
    double* z = sycl::malloc_shared<double>(size, q);
    sycl::event z_event = multiply_array_by_scalar(q, z_deps, size, t, vol_vol_two, z);
    if (sync)
        z_event.wait();

    //------------ c = quartnes * z ------------
    std::vector<sycl::event> c_deps;
    if (!sync)
        c_deps.push_back(z_event);
    double* c = sycl::malloc_shared<double>(size, q);
    sycl::event c_event = multiply_array_by_scalar(q, c_deps, size, z, 0.25, c);
    if (sync)
        c_event.wait();

    //------------ sqrt(z) ------------
    std::vector<sycl::event> sqrt_z_deps;
    if (!sync)
        sqrt_z_deps.push_back(z_event);
    double* sqrt_z = sycl::malloc_shared<double>(size, q);
    sycl::event sqrt_z_event = sqrt(q, sqrt_z_deps, size, z, sqrt_z);
    if (sync)
        sqrt_z_event.wait();
    
    sycl::free(z, q);

    //------------ y = ones / sqrt(z) ------------
    std::vector<sycl::event> y_deps;
    if (!sync)
        y_deps.push_back(sqrt_z_event);
    double* y = sycl::malloc_shared<double>(size, q);
    sycl::event y_event = divide_array_by_scalar(q, y_deps, size, 1, sqrt_z, y);
    if (sync)
        y_event.wait();

    sycl::free(sqrt_z, q);

    //------------ a_sub_b = a - b ------------
    std::vector<sycl::event> a_sub_b_deps;
    if (!sync) 
    {
        a_sub_b_deps.push_back(a_event);
        a_sub_b_deps.push_back(b_event);
    }
    double* a_sub_b = sycl::malloc_shared<double>(size, q);
    sycl::event a_sub_b_event = subtract(q, a_sub_b_deps, size, a, b, a_sub_b);
    if (sync)
        a_sub_b_event.wait();
    
    sycl::free(a, q);

    //------------ a_sub_b_add_c = a_sub_b + c ------------
    std::vector<sycl::event> a_sub_b_add_c_deps;
    if (!sync) 
    {
        a_sub_b_add_c_deps.push_back(c_event);
        a_sub_b_add_c_deps.push_back(a_sub_b_event);
    }
    double* a_sub_b_add_c = sycl::malloc_shared<double>(size, q);
    sycl::event a_sub_b_add_c_event = add(q, a_sub_b_add_c_deps, size, a_sub_b, c, a_sub_b_add_c);
    if (sync)
        a_sub_b_add_c_event.wait();

    //------------ w1 = a_sub_b_add_c * y ------------
    std::vector<sycl::event> w1_deps;
    if (!sync)
    {
        w1_deps.push_back(a_sub_b_add_c_event);
        w1_deps.push_back(y_event);
    }
    double* w1 = sycl::malloc_shared<double>(size, q);
    sycl::event w1_event = multiply(q, w1_deps, size, a_sub_b_add_c, y, w1);
    if (sync)
        w1_event.wait();
    
    sycl::free(a_sub_b_add_c, q);

     //------------ a_sub_b_sub_c = a_sub_b - c ------------
    std::vector<sycl::event> a_sub_b_sub_c_deps;
    if (!sync) 
    {
        a_sub_b_sub_c_deps.push_back(c_event);
        a_sub_b_sub_c_deps.push_back(a_sub_b_event);
    }
    double* a_sub_b_sub_c = sycl::malloc_shared<double>(size, q);
    sycl::event a_sub_b_sub_c_event = subtract(q, a_sub_b_sub_c_deps, size, a_sub_b, c, a_sub_b_sub_c);
    if (sync)
        a_sub_b_sub_c_event.wait();

    sycl::free(a_sub_b, q);
    sycl::free(c, q);

    //------------ w2 = a_sub_b_sub_c * y ------------
    std::vector<sycl::event> w2_deps;
    if (!sync) {
        w2_deps.push_back(a_sub_b_sub_c_event);
        w2_deps.push_back(y_event);
    }
    double* w2 = sycl::malloc_shared<double>(size, q);
    sycl::event w2_event = multiply(q, w2_deps, size, a_sub_b_sub_c, y, w2);
    if (sync)
        w2_event.wait();
    
    sycl::free(a_sub_b_sub_c, q);
    sycl::free(y, q);

    //------------ erf_w1 = erf(w1) ------------
    std::vector<sycl::event> erf_w1_deps;
    if (!sync) {
        erf_w1_deps.push_back(w1_event);
    }
    double* erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event erf_w1_event = erf(q, erf_w1_deps, size, w1, erf_w1);
    if (sync)
        erf_w1_event.wait();
    
    sycl::free(w1, q);

    //------------ halfs_mul_erf_w1 = halfs * erf_w1 ------------
    std::vector<sycl::event> halfs_mul_erf_w1_deps;
    if (!sync) {
        halfs_mul_erf_w1_deps.push_back(erf_w1_event);
    }
    double* halfs_mul_erf_w1 = sycl::malloc_shared<double>(size, q);
    sycl::event halfs_mul_erf_w1_event = multiply_array_by_scalar(q, halfs_mul_erf_w1_deps, size, erf_w1, 0.5, halfs_mul_erf_w1);
    if (sync)
        halfs_mul_erf_w1_event.wait();

    sycl::free(erf_w1, q);

    //------------ d1 = half + halfs_mul_erf_w1 ------------
    std::vector<sycl::event> d1_deps;
    if (!sync) {
        d1_deps.push_back(halfs_mul_erf_w1_event);
    }
    double* d1 = sycl::malloc_shared<double>(size, q);
    sycl::event d1_event = add_scalar_to_array(q, d1_deps, size, halfs_mul_erf_w1, 0.5, d1);
    if (sync)
        d1_event.wait();
    
    sycl::free(halfs_mul_erf_w1, q);

    //------------ erf_w2 = erf(w2) ------------
    std::vector<sycl::event> erf_w2_deps;
    if (!sync) {
        erf_w2_deps.push_back(w2_event);
    }
    double* erf_w2 = sycl::malloc_shared<double>(size, q);
    sycl::event erf_w2_event = erf(q, erf_w2_deps, size, w2, erf_w2);
    if (sync)
        erf_w2_event.wait();

    sycl::free(w2, q);

    //------------ halfs_mul_erf_w2 = half * erf_w2 ------------
    std::vector<sycl::event> halfs_mul_erf_w2_deps;
    if (!sync) {
        halfs_mul_erf_w2_deps.push_back(erf_w2_event);
    }
    double* halfs_mul_erf_w2 = sycl::malloc_shared<double>(size, q);
    sycl::event halfs_mul_erf_w2_event = multiply_array_by_scalar(q, halfs_mul_erf_w2_deps, size, erf_w2, 0.5, halfs_mul_erf_w2);
    if (sync)
        halfs_mul_erf_w2_event.wait();
    
    sycl::free(erf_w2, q);

    //------------ d2 = halfs + halfs_mul_erf_w2 ------------
    std::vector<sycl::event> d2_deps;
    if (!sync) {
        d2_deps.push_back(halfs_mul_erf_w2_event);
    }
    double* d2 = sycl::malloc_shared<double>(size, q);
    sycl::event d2_event = add_scalar_to_array(q, d2_deps, size, halfs_mul_erf_w2, 0.5, d2);
    if (sync)
        d2_event.wait();
    
    sycl::free(halfs_mul_erf_w2, q);

    //------------ exp_b = exp(b) ------------
    std::vector<sycl::event> exp_b_deps;
    if (!sync) {
        exp_b_deps.push_back(b_event);
    }
    double* exp_b = sycl::malloc_shared<double>(size, q);
    sycl::event exp_b_event = exp(q, exp_b_deps, size, b, exp_b);
    if (sync)
        exp_b_event.wait();
    
    sycl::free(b, q);

    //------------ Se = exp_b * strike ------------
    std::vector<sycl::event> Se_deps;
    if (!sync) {
        Se_deps.push_back(exp_b_event);
    }
    double* Se = sycl::malloc_shared<double>(size, q);
    sycl::event Se_event = multiply(q, Se_deps, size, exp_b, strike, Se);
    if (sync)
        Se_event.wait();
    
    sycl::free(exp_b, q);
    
    //------------ P_mul_d1 = price * d1 ------------
    std::vector<sycl::event> P_mul_d1_deps;
    if (!sync) {
        P_mul_d1_deps.push_back(d1_event);
    }
    double* P_mul_d1 = sycl::malloc_shared<double>(size, q);
    sycl::event P_mul_d1_event = multiply(q, P_mul_d1_deps, size, price, d1, P_mul_d1);
    if (sync)
        P_mul_d1_event.wait();
    
    sycl::free(d1, q);

    //------------ Se_mul_d2 = Se * d2 ------------
    std::vector<sycl::event> Se_mul_d2_deps;
    if (!sync) {
        Se_mul_d2_deps.push_back(Se_event);
        Se_mul_d2_deps.push_back(d2_event);
    }
    double* Se_mul_d2 = sycl::malloc_shared<double>(size, q);
    sycl::event Se_mul_d2_event = multiply(q, Se_mul_d2_deps, size, Se, d2, Se_mul_d2);
    if (sync)
        Se_mul_d2_event.wait();
    
    sycl::free(d2, q);

    //------------ r = P_mul_d1 - Se_mul_d2 ------------
    std::vector<sycl::event> r_deps;
    if (!sync) {
        r_deps.push_back(P_mul_d1_event);
        r_deps.push_back(Se_mul_d2_event);
    }
    double* r = sycl::malloc_shared<double>(size, q);
    sycl::event r_event = subtract(q, r_deps, size, P_mul_d1, Se_mul_d2, r);

    r_event.wait();

    memcpy(call, r, size*sizeof(double));

    sycl::free(P_mul_d1, q);
    sycl::free(Se_mul_d2, q);
    
    //------------ r_sub_P = r - P ------------
    std::vector<sycl::event> r_sub_P_deps;
    if (!sync) {
        r_sub_P_deps.push_back(r_event);
    }
    double* r_sub_P = sycl::malloc_shared<double>(size, q);
    sycl::event r_sub_P_event = subtract(q, r_sub_P_deps, size, call, price, r_sub_P);
    if (sync)
        r_sub_P_event.wait();

    //------------ r_sub_P_add_Se = r_sub_P + Se ------------
    std::vector<sycl::event> r_sub_P_add_Se_deps;
    if (!sync) {
        r_sub_P_add_Se_deps.push_back(r_event);
        r_sub_P_add_Se_deps.push_back(Se_event);
    }
    double* r_sub_P_add_Se = sycl::malloc_shared<double>(size, q);
    sycl::event r_sub_P_add_Se_event = add(q, r_sub_P_add_Se_deps, size, r_sub_P, Se, r_sub_P_add_Se);

    r_sub_P_add_Se_event.wait();

    memcpy(put, r_sub_P_add_Se, size*sizeof(double));
    
    sycl::free(Se, q);

}

double random(int a, int b) 
{
    return (double) rand() / (double) RAND_MAX * (b - a) + a;
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

    double *call = sycl::malloc_shared<double>(SIZE, q);
    double *put = sycl::malloc_shared<double>(SIZE, q);

    for (int i = 0; i < SIZE; i++)
    {
        price[i] = random(PL, PH); 
        strike[i] = random(SL, SH);
        t[i] =  random(TL, TH);
    }

    //------------ performanse mesure ------------
    std::vector<double> async_times;
    for (int i = 0; i < reprtitions; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE, q);

        auto t2 = std::chrono::high_resolution_clock::now();
        async_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count());
    }

    std::vector<double> sync_times;
    for (int i = 0; i < reprtitions; ++i) 
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE, q, true);

        auto t2 = std::chrono::high_resolution_clock::now();
        sync_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count());
    }

    std::cout << "Async time: " << median(async_times) << " s." << std::endl;
    std::cout << "Sync time: " << median(sync_times) << " s." << std::endl;
    std::cout << "Speedup: " << median(sync_times) / median(async_times) << "x" << std::endl;

    sycl::free(price, q);
    sycl::free(strike, q);
    sycl::free(t, q);
}