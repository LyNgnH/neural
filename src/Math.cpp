#pragma once
#include <cmath>
#include "Math.h"

namespace math_utils
{
	template <typename T>
	inline T sigmoid(T x)
    {
        return 1 / (1 + std::pow(e, -x));
    }
};
