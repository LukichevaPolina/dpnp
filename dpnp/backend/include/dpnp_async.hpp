#pragma once
#ifndef DPNP_ASYNC_H
#define DPNP_ASYNC_H

#include <memory>

class Deps
{
private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
public:
  Impl* get_pImpl();
  void add(Deps*);
  void wait();
  Deps();
  ~Deps();
};

#endif // DPNP_ASYNC_H
