// #include <iostream>

#include "dpnp_async.hpp"
#include "dpnp_async_pimpl.hpp"

Deps::Impl* Deps::get_pImpl() { return pImpl.get(); }
void Deps::add(Deps* d) {pImpl->add(d->get_pImpl()->get()); }
void Deps::wait() {pImpl->wait();}
Deps::Deps() : pImpl(std::make_unique<Impl>()) {} 
Deps::~Deps() {}