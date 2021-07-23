#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

#ifdef ITT

#include <ittnotify.h>

__itt_domain* domain_bs = __itt_domain_create("bs");

__itt_string_handle* handle_malloc_price = __itt_string_handle_create("malloc_price");
__itt_string_handle* handle_malloc_strike = __itt_string_handle_create("malloc_strike");
__itt_string_handle* handle_malloc_t = __itt_string_handle_create("malloc_t");
__itt_string_handle* handle_malloc_call = __itt_string_handle_create("malloc_call");
__itt_string_handle* handle_malloc_put = __itt_string_handle_create("malloc_put");
__itt_string_handle* handle_malloc_p_div_s = __itt_string_handle_create("malloc_p_div_s");
__itt_string_handle* handle_malloc_a = __itt_string_handle_create("malloc_a");
__itt_string_handle* handle_malloc_b = __itt_string_handle_create("malloc_b");
__itt_string_handle* handle_malloc_z = __itt_string_handle_create("malloc_z");
__itt_string_handle* handle_malloc_c = __itt_string_handle_create("malloc_c");
__itt_string_handle* handle_malloc_sqrt_z = __itt_string_handle_create("malloc_sqrt_z");
__itt_string_handle* handle_malloc_y = __itt_string_handle_create("malloc_y");
__itt_string_handle* handle_malloc_a_sub_b = __itt_string_handle_create("malloc_a_sub_b");
__itt_string_handle* handle_malloc_a_sub_b_add_c = __itt_string_handle_create("malloc_a_sub_b_add_c");
__itt_string_handle* handle_malloc_w1 = __itt_string_handle_create("malloc_w1");
__itt_string_handle* handle_malloc_a_sub_b_sub_c = __itt_string_handle_create("malloc_a_sub_b_sub_c");
__itt_string_handle* handle_malloc_w2 = __itt_string_handle_create("malloc_w2");
__itt_string_handle* handle_malloc_erf_w1 = __itt_string_handle_create("malloc_erf_w1");
__itt_string_handle* handle_malloc_halfs_mul_erf_w1 = __itt_string_handle_create("malloc_halfs_mul_erf_w1");
__itt_string_handle* handle_malloc_d1 = __itt_string_handle_create("malloc_d1");
__itt_string_handle* handle_malloc_erf_w2 = __itt_string_handle_create("malloc_erf_w2");
__itt_string_handle* handle_malloc_halfs_mul_erf_w2 = __itt_string_handle_create("malloc_halfs_mul_erf_w2");
__itt_string_handle* handle_malloc_d2= __itt_string_handle_create("malloc_d2");
__itt_string_handle* handle_malloc_exp_b = __itt_string_handle_create("malloc_exp_b");
__itt_string_handle* handle_malloc_Se = __itt_string_handle_create("malloc_Se");
__itt_string_handle* handle_malloc_P_mul_d1 = __itt_string_handle_create("malloc_P_mul_d1");
__itt_string_handle* handle_malloc_Se_mul_d2 = __itt_string_handle_create("malloc_Se_mul_d2");
__itt_string_handle* handle_malloc_r = __itt_string_handle_create("malloc_r");
__itt_string_handle* handle_malloc_r_sub_P = __itt_string_handle_create("malloc_r_sub_P");
__itt_string_handle* handle_malloc_r_sub_P_add_Se = __itt_string_handle_create("malloc_r_sub_P_add_Se");

__itt_string_handle* handle_submit_p_div_s = __itt_string_handle_create("submit_p_div_s");
__itt_string_handle* handle_submit_a = __itt_string_handle_create("submit_a");
__itt_string_handle* handle_submit_b = __itt_string_handle_create("submit_b");
__itt_string_handle* handle_submit_z = __itt_string_handle_create("submit_z");
__itt_string_handle* handle_submit_c = __itt_string_handle_create("submit_c");
__itt_string_handle* handle_submit_sqrt_z = __itt_string_handle_create("submit_sqrt_z");
__itt_string_handle* handle_submit_y = __itt_string_handle_create("submit_y");
__itt_string_handle* handle_submit_a_sub_b = __itt_string_handle_create("submit_a_sub_b");
__itt_string_handle* handle_submit_a_sub_b_add_c = __itt_string_handle_create("submit_a_sub_b_add_c");
__itt_string_handle* handle_submit_w1 = __itt_string_handle_create("submit_w1");
__itt_string_handle* handle_submit_a_sub_b_sub_c = __itt_string_handle_create("submit_a_sub_b_sub_c");
__itt_string_handle* handle_submit_w2 = __itt_string_handle_create("submit_w2");
__itt_string_handle* handle_submit_erf_w1 = __itt_string_handle_create("submit_erf_w1");
__itt_string_handle* handle_submit_halfs_mul_erf_w1 = __itt_string_handle_create("submit_halfs_mul_erf_w1");
__itt_string_handle* handle_submit_d1 = __itt_string_handle_create("submit_d1");
__itt_string_handle* handle_submit_erf_w2 = __itt_string_handle_create("submit_erf_w2");
__itt_string_handle* handle_submit_halfs_mul_erf_w2 = __itt_string_handle_create("submit_halfs_mul_erf_w2");
__itt_string_handle* handle_submit_d2= __itt_string_handle_create("submit_d2");
__itt_string_handle* handle_submit_exp_b = __itt_string_handle_create("submit_exp_b");
__itt_string_handle* handle_submit_Se = __itt_string_handle_create("submit_Se");
__itt_string_handle* handle_submit_P_mul_d1 = __itt_string_handle_create("submit_P_mul_d1");
__itt_string_handle* handle_submit_Se_mul_d2 = __itt_string_handle_create("submit_Se_mul_d2");
__itt_string_handle* handle_submit_r = __itt_string_handle_create("submit_r");
__itt_string_handle* handle_submit_r_sub_P = __itt_string_handle_create("submit_r_sub_P");
__itt_string_handle* handle_submit_r_sub_P_add_Se = __itt_string_handle_create("submit_r_sub_P_add_Se");

__itt_string_handle* handle_wait_p_div_s = __itt_string_handle_create("wait_p_div_s");
__itt_string_handle* handle_wait_a = __itt_string_handle_create("wait_a");
__itt_string_handle* handle_wait_b = __itt_string_handle_create("wait_b");
__itt_string_handle* handle_wait_z = __itt_string_handle_create("wait_z");
__itt_string_handle* handle_wait_c = __itt_string_handle_create("wait_c");
__itt_string_handle* handle_wait_sqrt_z = __itt_string_handle_create("wait_sqrt_z");
__itt_string_handle* handle_wait_y = __itt_string_handle_create("wait_y");
__itt_string_handle* handle_wait_a_sub_b = __itt_string_handle_create("wait_a_sub_b");
__itt_string_handle* handle_wait_a_sub_b_add_c = __itt_string_handle_create("wait_a_sub_b_add_c");
__itt_string_handle* handle_wait_w1 = __itt_string_handle_create("wait_w1");
__itt_string_handle* handle_wait_a_sub_b_sub_c = __itt_string_handle_create("wait_a_sub_b_sub_c");
__itt_string_handle* handle_wait_w2 = __itt_string_handle_create("wait_w2");
__itt_string_handle* handle_wait_erf_w1 = __itt_string_handle_create("wait_erf_w1");
__itt_string_handle* handle_wait_halfs_mul_erf_w1 = __itt_string_handle_create("wait_halfs_mul_erf_w1");
__itt_string_handle* handle_wait_d1 = __itt_string_handle_create("wait_d1");
__itt_string_handle* handle_wait_erf_w2 = __itt_string_handle_create("wait_erf_w2");
__itt_string_handle* handle_wait_halfs_mul_erf_w2 = __itt_string_handle_create("wait_halfs_mul_erf_w2");
__itt_string_handle* handle_wait_d2= __itt_string_handle_create("wait_d2");
__itt_string_handle* handle_wait_exp_b = __itt_string_handle_create("wait_exp_b");
__itt_string_handle* handle_wait_Se = __itt_string_handle_create("wait_Se");
__itt_string_handle* handle_wait_P_mul_d1 = __itt_string_handle_create("wait_P_mul_d1");
__itt_string_handle* handle_wait_Se_mul_d2 = __itt_string_handle_create("wait_Se_mul_d2");
__itt_string_handle* handle_wait_r = __itt_string_handle_create("wait_r");
__itt_string_handle* handle_wait_r_sub_P = __itt_string_handle_create("wait_r_sub_P");
__itt_string_handle* handle_wait_r_sub_P_add_Se = __itt_string_handle_create("wait_r_sub_P_add_Se");
__itt_string_handle* handle_wait_call = __itt_string_handle_create("wait_call");
__itt_string_handle* handle_wait_put = __itt_string_handle_create("wait_put");

__itt_string_handle* handle_memcpy_r = __itt_string_handle_create("memcpy_r");
__itt_string_handle* handle_memcpy_r_sub_P_add_Se = __itt_string_handle_create("memcpy_r_sub_P_add_Se");

__itt_string_handle* handle_init_price = __itt_string_handle_create("init_price");
__itt_string_handle* handle_init_strike = __itt_string_handle_create("init_strike");
__itt_string_handle* handle_init_t = __itt_string_handle_create("init_t");

__itt_string_handle* handle_warming_up = __itt_string_handle_create("warming_up");

__itt_string_handle* handle_async_run = __itt_string_handle_create("async_run");
__itt_string_handle* handle_sync_run = __itt_string_handle_create("sync_run");


#define itt_task_begin(handle) __itt_task_begin(domain_bs, __itt_null, __itt_null, handle)
#define itt_task_end __itt_task_end(domain_bs)

#else
#define itt_task_begin(handle)
#define itt_task_end

#endif

sycl::event divide(sycl::queue q,
                   std::vector<sycl::event> &deps,
                   const size_t size,
                   double* a,
                   double* b,
                   double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
        cgh.depends_on(deps);
        cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = a[i] / b[i];}); 
    });

    return event;  
}

sycl::event divide_scalar_by_array(sycl::queue q,
                                   std::vector<sycl::event> &deps,
                                   const size_t size,
                                   double a,
                                   double* b,
                                   double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh)
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh)
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh)
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
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
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
        cgh.depends_on(deps);
        cgh.parallel_for(size, [=](sycl::item<1> i) { y[i] = sycl::exp(a[i]);});
    });

    return event;
}

sycl::event copy(sycl::queue q,
                 std::vector<sycl::event> &deps,
                 const size_t size,
                 double* a,
                 double* y)
{
    sycl::event event = q.submit([&](sycl::handler &cgh) 
    {
        cgh.depends_on(deps);
        cgh.memcpy(y, a, size*sizeof(double));
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

    itt_task_begin(handle_malloc_p_div_s);
    double* p_div_s = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> p_div_s_deps;

    itt_task_begin(handle_submit_p_div_s);
    sycl::event p_div_s_event = divide(q, p_div_s_deps, size, price, strike, p_div_s);
    itt_task_end;

    if (sync) 
    {
        itt_task_begin(handle_wait_p_div_s);
        p_div_s_event.wait();
        itt_task_end;
    }

    // ------------ a = log(p_div_s) ------------
    itt_task_begin(handle_malloc_a);
    double* a = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> a_deps;
    if (!sync) 
    {
        a_deps.push_back(p_div_s_event);
    }

    itt_task_begin(handle_submit_a);
    sycl::event a_event = log(q, a_deps, size, p_div_s, a);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_a);
        a_event.wait();
        itt_task_end;
    }

    // ------------ b = t * mr ------------
    itt_task_begin(handle_malloc_b);
    double* b = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> b_deps;

    itt_task_begin(handle_submit_b);
    sycl::event b_event = multiply_array_by_scalar(q, b_deps, size, t, mr, b);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_b);
        b_event.wait();
        itt_task_end;
    }

    //------------ z = t * vol_vol_two ------------
    itt_task_begin(handle_malloc_z);
    double* z = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> z_deps;

    itt_task_begin(handle_submit_z);
    sycl::event z_event = multiply_array_by_scalar(q, z_deps, size, t, vol_vol_two, z);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_z);
        z_event.wait();
        itt_task_end;
    }

    //------------ c = 0.25 * z ------------
    itt_task_begin(handle_malloc_c);
    double* c = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> c_deps;
    if (!sync)
        c_deps.push_back(z_event);

    itt_task_begin(handle_submit_c);
    sycl::event c_event = multiply_array_by_scalar(q, c_deps, size, z, 0.25, c);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_c);
        c_event.wait();
        itt_task_end;
    }

    //------------ sqrt_z = sqrt(z) ------------
    itt_task_begin(handle_malloc_sqrt_z);
    double* sqrt_z = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> sqrt_z_deps;
    if (!sync)
        sqrt_z_deps.push_back(z_event);
    
    itt_task_begin(handle_submit_sqrt_z);
    sycl::event sqrt_z_event = sqrt(q, sqrt_z_deps, size, z, sqrt_z);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_sqrt_z);
        sqrt_z_event.wait();
        itt_task_end;
    }

    //------------ y = 1 / sqrt(z) ------------
    itt_task_begin(handle_malloc_y);
    double* y = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> y_deps;
    if (!sync)
        y_deps.push_back(sqrt_z_event);

    itt_task_begin(handle_submit_y);
    sycl::event y_event = divide_scalar_by_array(q, y_deps, size, 1, sqrt_z, y);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_y);
        y_event.wait();
        itt_task_end;
    }

    //------------ a_sub_b = a - b ------------
    itt_task_begin(handle_malloc_a_sub_b);
    double* a_sub_b = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> a_sub_b_deps;
    if (!sync) 
    {
        a_sub_b_deps.push_back(a_event);
        a_sub_b_deps.push_back(b_event);
    }

    itt_task_begin(handle_submit_a_sub_b);
    sycl::event a_sub_b_event = subtract(q, a_sub_b_deps, size, a, b, a_sub_b);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_a_sub_b);
        a_sub_b_event.wait();
        itt_task_end;
    }

    //------------ a_sub_b_add_c = a_sub_b + c ------------
    itt_task_begin(handle_malloc_a_sub_b_add_c);
    double* a_sub_b_add_c = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> a_sub_b_add_c_deps;
    if (!sync) 
    {
        a_sub_b_add_c_deps.push_back(c_event);
        a_sub_b_add_c_deps.push_back(a_sub_b_event);
    }

    itt_task_begin(handle_submit_a_sub_b_add_c);
    sycl::event a_sub_b_add_c_event = add(q, a_sub_b_add_c_deps, size, a_sub_b, c, a_sub_b_add_c);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_a_sub_b_add_c);
        a_sub_b_add_c_event.wait();
        itt_task_end;
    }

    //------------ w1 = a_sub_b_add_c * y ------------
    itt_task_begin(handle_malloc_w1);
    double* w1 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> w1_deps;
    if (!sync)
    {
        w1_deps.push_back(a_sub_b_add_c_event);
        w1_deps.push_back(y_event);
    }

    itt_task_begin(handle_submit_w1);
    sycl::event w1_event = multiply(q, w1_deps, size, a_sub_b_add_c, y, w1);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_w1);
        w1_event.wait();
        itt_task_end;
    }

     //------------ a_sub_b_sub_c = a_sub_b - c ------------
    itt_task_begin(handle_malloc_a_sub_b_sub_c);
    double* a_sub_b_sub_c = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> a_sub_b_sub_c_deps;
    if (!sync) 
    {
        a_sub_b_sub_c_deps.push_back(c_event);
        a_sub_b_sub_c_deps.push_back(a_sub_b_event);
    }

    itt_task_begin(handle_submit_a_sub_b_sub_c);
    sycl::event a_sub_b_sub_c_event = subtract(q, a_sub_b_sub_c_deps, size, a_sub_b, c, a_sub_b_sub_c);
    itt_task_end;

    if (sync)
    {   
        itt_task_begin(handle_wait_a_sub_b_sub_c);
        a_sub_b_sub_c_event.wait();        
        itt_task_end;
    }

    //------------ w2 = a_sub_b_sub_c * y ------------
    itt_task_begin(handle_malloc_w2);
    double* w2 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

     std::vector<sycl::event> w2_deps;
    if (!sync) 
    {
        w2_deps.push_back(a_sub_b_sub_c_event);
        w2_deps.push_back(y_event);
    }

    itt_task_begin(handle_submit_w2);
    sycl::event w2_event = multiply(q, w2_deps, size, a_sub_b_sub_c, y, w2);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_w2);
        w2_event.wait();
        itt_task_end;
    }

    //------------ erf_w1 = erf(w1) ------------
    itt_task_begin(handle_malloc_erf_w1);
    double* erf_w1 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> erf_w1_deps;
    if (!sync) 
        erf_w1_deps.push_back(w1_event);

    itt_task_begin(handle_submit_erf_w1);
    sycl::event erf_w1_event = erf(q, erf_w1_deps, size, w1, erf_w1);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_erf_w1);
        erf_w1_event.wait();
        itt_task_end;
    }

    //------------ halfs_mul_erf_w1 = 0.5 * erf_w1 ------------
    itt_task_begin(handle_malloc_halfs_mul_erf_w1);
    double* halfs_mul_erf_w1 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> halfs_mul_erf_w1_deps;
    if (!sync) 
        halfs_mul_erf_w1_deps.push_back(erf_w1_event);

    itt_task_begin(handle_submit_halfs_mul_erf_w1);
    sycl::event halfs_mul_erf_w1_event = multiply_array_by_scalar(q, halfs_mul_erf_w1_deps, size, erf_w1, 0.5, halfs_mul_erf_w1);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_halfs_mul_erf_w1);
        halfs_mul_erf_w1_event.wait();
        itt_task_end;
    }

    //------------ d1 = 0.5 + halfs_mul_erf_w1 ------------
    itt_task_begin(handle_malloc_d1);
    double* d1 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> d1_deps;
    if (!sync) 
        d1_deps.push_back(halfs_mul_erf_w1_event);

    itt_task_begin(handle_submit_d1);
    sycl::event d1_event = add_scalar_to_array(q, d1_deps, size, halfs_mul_erf_w1, 0.5, d1);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_d1);
        d1_event.wait();
        itt_task_end;
    }

    //------------ erf_w2 = erf(w2) ------------
    itt_task_begin(handle_malloc_erf_w2);
    double* erf_w2 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> erf_w2_deps;
    if (!sync) 
        erf_w2_deps.push_back(w2_event);


    itt_task_begin(handle_submit_erf_w2);
    sycl::event erf_w2_event = erf(q,erf_w2_deps, size, w2, erf_w2);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_erf_w2);
        erf_w2_event.wait();
        itt_task_end;
    }

    //------------ halfs_mul_erf_w2 = 0.5 * erf_w2 ------------
    itt_task_begin(handle_malloc_halfs_mul_erf_w2);
    double* halfs_mul_erf_w2 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> halfs_mul_erf_w2_deps;
    if (!sync) 
        halfs_mul_erf_w2_deps.push_back(erf_w2_event);

    itt_task_begin(handle_submit_halfs_mul_erf_w2);
    sycl::event halfs_mul_erf_w2_event = multiply_array_by_scalar(q, halfs_mul_erf_w2_deps, size, erf_w2, 0.5, halfs_mul_erf_w2);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_halfs_mul_erf_w2);
        halfs_mul_erf_w2_event.wait();
        itt_task_end;
    }

    //------------ d2 = 0.5 + halfs_mul_erf_w2 ------------
    itt_task_begin(handle_malloc_d2);
    double* d2 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> d2_deps;
    if (!sync) 
        d2_deps.push_back(halfs_mul_erf_w2_event);

    itt_task_begin(handle_submit_d2);
    sycl::event d2_event = add_scalar_to_array(q, d2_deps, size, halfs_mul_erf_w2, 0.5, d2);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_d2);
        d2_event.wait();
        itt_task_end;
    }

    //------------ exp_b = exp(b) ------------
    itt_task_begin(handle_malloc_exp_b);
    double* exp_b = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> exp_b_deps;
    if (!sync) 
        exp_b_deps.push_back(b_event);

    itt_task_begin(handle_submit_exp_b);
    sycl::event exp_b_event = exp(q, exp_b_deps, size, b, exp_b);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_exp_b);
        exp_b_event.wait();
        itt_task_end;
    }

    //------------ Se = exp_b * strike ------------
    itt_task_begin(handle_malloc_Se);
    double* Se = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> Se_deps;
    if (!sync) 
        Se_deps.push_back(exp_b_event);

    itt_task_begin(handle_submit_Se);
    sycl::event Se_event = multiply(q, Se_deps, size, exp_b, strike, Se);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_Se);
        Se_event.wait();
        itt_task_end;
    }

    //------------ P_mul_d1 = price * d1 ------------
    itt_task_begin(handle_malloc_P_mul_d1);
    double* P_mul_d1 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> P_mul_d1_deps;
    if (!sync) 
        P_mul_d1_deps.push_back(d1_event);

    itt_task_begin(handle_submit_P_mul_d1);
    sycl::event P_mul_d1_event = multiply(q, P_mul_d1_deps, size, price, d1, P_mul_d1);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_P_mul_d1);
        P_mul_d1_event.wait();
        itt_task_end;
    }

    //------------ Se_mul_d2 = Se * d2 ------------
    itt_task_begin(handle_malloc_Se_mul_d2);
    double* Se_mul_d2 = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> Se_mul_d2_deps;
    if (!sync) 
    {
        Se_mul_d2_deps.push_back(Se_event);
        Se_mul_d2_deps.push_back(d2_event);
    }

    itt_task_begin(handle_submit_Se_mul_d2);
    sycl::event Se_mul_d2_event = multiply(q, Se_mul_d2_deps, size, Se, d2, Se_mul_d2);
    itt_task_end;

    if (sync)
    {
        itt_task_begin(handle_wait_Se_mul_d2);
        Se_mul_d2_event.wait();
        itt_task_end;
    }

    //------------ r = P_mul_d1 - Se_mul_d2 ------------
    itt_task_begin(handle_malloc_r);
    double* r = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> r_deps;
    if (!sync) 
    {
        r_deps.push_back(P_mul_d1_event);
        r_deps.push_back(Se_mul_d2_event);
    }

    itt_task_begin(handle_submit_r);
    sycl::event r_event = subtract(q, r_deps, size, P_mul_d1, Se_mul_d2, r);
    itt_task_end;
    
    if (sync)
    {
        itt_task_begin(handle_wait_r);
        r_event.wait();
        itt_task_end;
    }

    std::vector<sycl::event> call_deps;

    if (!sync)
        call_deps.push_back(r_event);

    itt_task_begin(handle_memcpy_r);
    sycl::event  call_event = copy(q, call_deps, size, r, call);
    itt_task_end;

    //------------ r_sub_P = r - P ------------
    itt_task_begin(handle_malloc_r_sub_P);
    double* r_sub_P = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> r_sub_P_deps;
    if (!sync) 
        r_sub_P_deps.push_back(r_event);

    itt_task_begin(handle_submit_r_sub_P);
    sycl::event r_sub_P_event = subtract(q, r_sub_P_deps, size, call, price, r_sub_P);
    itt_task_end;

    if (sync)
    {   
        itt_task_begin(handle_wait_r_sub_P);
        r_sub_P_event.wait();
        itt_task_end;
    }

    //------------ r_sub_P_add_Se = r_sub_P + Se ------------
    itt_task_begin(handle_malloc_r_sub_P_add_Se);
    double* r_sub_P_add_Se = sycl::malloc_shared<double>(size, q);
    itt_task_end;

    std::vector<sycl::event> r_sub_P_add_Se_deps;
    if (!sync) 
    {
        r_sub_P_add_Se_deps.push_back(r_event);
        r_sub_P_add_Se_deps.push_back(Se_event);
    }

    itt_task_begin(handle_submit_r_sub_P_add_Se);
    sycl::event r_sub_P_add_Se_event = add(q, r_sub_P_add_Se_deps, size, r_sub_P, Se, r_sub_P_add_Se);
    itt_task_end;
    if (sync)
    {
        itt_task_begin(handle_wait_r_sub_P_add_Se);
        r_sub_P_add_Se_event.wait();
        itt_task_end;
    }

    std::vector<sycl::event> put_deps;
    if (!sync)
        put_deps.push_back(r_sub_P_add_Se_event);

    itt_task_begin(handle_memcpy_r_sub_P_add_Se);
    sycl::event put_event = copy(q, put_deps, size, r_sub_P_add_Se, put);
    itt_task_end;

    itt_task_begin(handle_wait_call);
    call_event.wait();
    itt_task_end;

    itt_task_begin(handle_wait_put);
    put_event.wait();
    itt_task_end;

    sycl::free(p_div_s, q);
    sycl::free(z, q);
    sycl::free(sqrt_z, q);
    sycl::free(a, q);
    sycl::free(a_sub_b_add_c, q);
    sycl::free(a_sub_b, q);
    sycl::free(c, q);
    sycl::free(a_sub_b_sub_c, q);
    sycl::free(y, q);
    sycl::free(w1, q);
    sycl::free(erf_w1, q);
    sycl::free(halfs_mul_erf_w1, q);
    sycl::free(w2, q);
    sycl::free(erf_w2, q);
    sycl::free(halfs_mul_erf_w2, q);
    sycl::free(b, q);
    sycl::free(exp_b, q);
    sycl::free(d1, q);
    sycl::free(d2, q);
    sycl::free(P_mul_d1, q);
    sycl::free(Se_mul_d2, q);
    sycl::free(Se, q);
    sycl::free(r, q);
    sycl::free(r_sub_P, q);
    sycl::free(r_sub_P_add_Se, q);
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

int main(int argc, char *argv[]) {

    int repetitions = 1;
    if (argc > 2)
        repetitions = std::stoi(argv[2]);

    size_t SIZE = 20;
    if (argc > 1)
        SIZE = std::stoi(argv[1]);

    const size_t SEED = 7777777;
    const long PL = 10, PH = 50;
    const long SL = 10, SH = 50;
    const long TL = 1, TH = 2;
    const double RISK_FREE = 0.1;
    const double VOLATILITY = 0.2;

    sycl::queue q{sycl::gpu_selector{}};

    itt_task_begin(handle_malloc_price);
    double *price = sycl::malloc_shared<double>(SIZE, q);
    itt_task_end;

    itt_task_begin(handle_malloc_strike);
    double *strike = sycl::malloc_shared<double>(SIZE, q);
    itt_task_end;

    itt_task_begin(handle_malloc_t);
    double *t = sycl::malloc_shared<double>(SIZE, q);
    itt_task_end;

    itt_task_begin(handle_malloc_call);
    double *call = sycl::malloc_shared<double>(SIZE, q);
    itt_task_end;

    itt_task_begin(handle_malloc_put);
    double *put = sycl::malloc_shared<double>(SIZE, q);
    itt_task_end;

    //------------ initialization ------------
    itt_task_begin(handle_init_price);
    for (int i = 0; i < SIZE; i++)
        price[i] = random(PL, PH); 
    itt_task_end;

    itt_task_begin(handle_init_strike);
    for (int i = 0; i < SIZE; i++)
        strike[i] = random(SL, SH);
    itt_task_end;

    itt_task_begin(handle_init_t);
    for (int i = 0; i < SIZE; i++)
        t[i] =  random(TL, TH);
    itt_task_end;

    //------------ warming up ------------
    itt_task_begin(handle_warming_up);
    black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE, q);
    itt_task_end;

    //------------ performance mesure ------------
    std::vector<double> async_times;
    for (int i = 0; i < repetitions; ++i) 
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        itt_task_begin(handle_async_run);
        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE, q);
        itt_task_end;
        auto t2 = std::chrono::high_resolution_clock::now();
        async_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count());
    }

    std::vector<double> sync_times;
    for (int i = 0; i < repetitions; ++i)  
    {   
        auto t1 = std::chrono::high_resolution_clock::now();
        itt_task_begin(handle_sync_run);
        black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put, SIZE, q, true);
        itt_task_end;
        auto t2 = std::chrono::high_resolution_clock::now();
        sync_times.push_back(std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count());
    }

    std::cout << std::endl;
    std::cout << "Async time: " << median(async_times) << " s." << std::endl;
    std::cout << "Sync time: " <<  median(sync_times) << " s." << std::endl;
    std::cout << "Speedup: " <<  median(sync_times) /  median(async_times) << "x" << std::endl;

    sycl::free(price, q);
    sycl::free(strike, q);
    sycl::free(t, q);
    sycl::free(call, q);
    sycl::free(put, q);
}
