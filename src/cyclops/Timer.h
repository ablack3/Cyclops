/*
 * Timer.h
 *
 *  Created on: Apr 25, 2014
 *      Author: msuchard
 */

#ifndef TIMER_H_
#define TIMER_H_

#include "Timing.h"

namespace bsccs {

// Elapsed wall-clock seconds since construction.
//
// Previously built on gettimeofday() and <sys/time.h>, which MSVC does not
// provide. bsccs::chrono is portable and monotonic.
class Timer {
public:
	Timer();

	double operator()();

	virtual ~Timer();

private:
	bsccs::chrono::steady_clock::time_point time1;
};

} /* namespace bsccs */
#endif /* TIMER_H_ */
