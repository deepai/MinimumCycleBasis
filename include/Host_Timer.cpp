#include "Host_Timer.h"

double HostTimer::start_timer() {
	elapsedTime = omp_get_wtime();
	return elapsedTime;
}

double HostTimer::stop_timer() {
	double totalTimeElapsed = omp_get_wtime();

	return totalTimeElapsed;
}
