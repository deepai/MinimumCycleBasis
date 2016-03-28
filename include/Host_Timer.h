#ifndef __HOST_TIMER
#define __HOST_TIMER

#include "omp.h"

class HostTimer
{
	double elapsedTime;

public:

	double start_timer();
	double stop_timer();

};

#endif