/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef __MicroURNG_dot_hpp__
#define __MicroURNG_dot_hpp__

#include <stdexcept>
#include <limits>

namespace r123{
/**
    Given a CBRNG whose ctr_type has an unsigned integral value_type,
    MicroURNG<CBRNG, BITS>(c, k) is a type that satisfies the
    requirements of a C++0x Uniform Random Number Generator.

    The intended purpose is for a MicroURNG to be passed
    as an argument to a C++0x Distribution, e.g.,
    std::normal_distribution.  See examples/MicroURNG.cpp.

    The MicroURNG functor may only be called a limited number of
    times.  After

       ctr_type.size()*2^BITS,

    calls, it throws a runtime_error.

    The high BITS bits of the highest word in the counter c, passed to
    the constructor must be zero.  MicroURNG uses these bits to
    "count".

    Notice that you can carve out as many BITS as you like from the
    counter space, and are then free to call the MicroURNG
    until the bits run out.  If you want a few dozen normally
    distributed randoms for each of your state/timestep/seed
    tuples, then give MicroURNG a few extra bits to work with and
    you're good to go:

\code
       typedef ?someCBRNG? RNG;
       RNG::ctr_type c = ...; // under application control
       RNG::key_type k = ...; // 
       std::normal_distribution<float> nd;
       MicroURNG<RNG, 10> urng(c, k);
       for(???){
         ...
         nd(urng);  // may be called several hundred times with BITS=10
         ...
       }
\endcode
*/

template<typename CBRNG, unsigned int _BITS>
class MicroURNG{
    // According to C++0x, a URNG requires only a result_type,
    // operator()(), min() and max() methods.  Everything else
    // (ctr_type, key_type, reset() method, etc.) is "value added"
    // for the benefit of users that "know" that they're dealing with
    // a MicroURNG.
public:
    typedef CBRNG cbrng_type;
    static const unsigned BITS = _BITS;
    typedef typename cbrng_type::ctr_type ctr_type;
    typedef typename cbrng_type::key_type key_type;
    typedef typename cbrng_type::ukey_type ukey_type;
    typedef typename ctr_type::value_type result_type;

    result_type operator()(){
        if(last_elem == 0){
            // jam n into the high bits of c
            if( n >= (((R123_ULONG_LONG)1)<<BITS) )
                throw std::runtime_error("Incremented high bits of MicroURNG's counter too many times");
            const size_t W = std::numeric_limits<result_type>::digits;
            ctr_type c = c0;
            c[c0.size()-1] |= n<<(W-BITS);
            rdata = b(c,k);
            n++;
            last_elem = rdata.size();
        }
        return rdata[--last_elem];
    }
    MicroURNG(cbrng_type _b, ctr_type _c0, ukey_type _uk) : b(_b), c0(_c0), k(_uk), n(0), last_elem(0) {
        chkhighbits();
    }
    MicroURNG(ctr_type _c0, ukey_type _uk) : b(), c0(_c0), k(_uk), n(0), last_elem(0) {
        chkhighbits();
    }
    result_type min() const{
        return std::numeric_limits<result_type>::min();
    }
    result_type max() const{
        return std::numeric_limits<result_type>::max();
    }
    // extra methods:
    const ctr_type& counter() const{ return c0; }
    void reset(ctr_type _c0, ukey_type _uk){
        c0 = _c0;
        chkhighbits();
        k = _uk;
        n = 0;
        last_elem = 0;
    }

private:
    cbrng_type b;
    ctr_type c0;
    key_type k;
    R123_ULONG_LONG n;
    size_t last_elem;
    ctr_type rdata;
    void chkhighbits(){
        result_type r = c0[c0.size()-1];
        result_type mask = std::numeric_limits<result_type>::max()>>BITS;
        if((r&mask) != r)
            throw std::runtime_error("MicroURNG: c0, does not have high bits clear");
    }
};
} // namespace r123
#endif
