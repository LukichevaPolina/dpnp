#pragma once
#ifndef DPNP_ASYNC_PIMPL_H
#define DPNP_ASYNC_PIMPL_H

#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

#include "dpnp_async.hpp"

class Deps::Impl
{
private:
  std::vector<sycl::event> deps;
public:
  std::vector<sycl::event> get() {return deps;}
  void add(sycl::event e) { deps.push_back(e);}
  void add(std::vector<sycl::event> d) { deps.insert(deps.end(), d.begin(), d.end());}
  void wait() {sycl::event::wait(deps);}
  Impl() : deps({}) {}; 
  ~Impl() 
  {
    sycl::event::wait(deps);
    deps.clear();
  }
};

#endif // DPNP_ASYNC_PIMPL_H