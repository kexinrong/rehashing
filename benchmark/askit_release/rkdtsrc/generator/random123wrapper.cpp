#include <stdlib.h>
#include "threefry.h"
#include "random123wrapper.h"

// simply incrementing the counter (or key) is effectively indistinguishable from a sequence
// of samples of a uniformly distributed random variable
int rand123(int counter)
{
	typedef r123::Threefry2x64 CBRNG;
	CBRNG rgenerator;
	CBRNG::ctr_type ctr={{0, 0}};
	CBRNG::key_type key={{0xadeafbee, 0xdeedbeef}};

	ctr[0] = counter;
	ctr[1] = counter*counter;

	CBRNG::ctr_type rand = rgenerator(ctr, key);
	
	return rand[0]%R123_MAX + 1;

}


int rand123()
{
	typedef r123::Threefry2x64 CBRNG;
	CBRNG rgenerator;
	CBRNG::ctr_type ctr={{0, 0}};
	CBRNG::key_type key={{0xadeafbee, 0xdeedbeef}};

	ctr[0] = rand();
	ctr[1] = rand();

	CBRNG::ctr_type rand = rgenerator(ctr, key);
	
	return rand[0]%R123_MAX + 1;

}



