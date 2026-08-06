/*
 * Timer.cpp
 *
 *  Created on: Apr 25, 2014
 *      Author: msuchard
 */

#include "Timer.h"

namespace bsccs {

Timer::Timer() : time1(bsccs::chrono::steady_clock::now()) { }

double Timer::operator()() {
	const auto time2 = bsccs::chrono::steady_clock::now();
	return bsccs::chrono::duration<double>(time2 - time1).count();
}

Timer::~Timer() { }

} /* namespace bsccs */
