#include <chrono>

using namespace std;

double convertToDouble(int value)
{
    return (double)value;
};

double convertToDouble(double value)
{
    return value;
};

double convertToDouble(long value)
{
    return (double)value;
};

template <typename T>
T convertFromDouble(double value)
{
    //throw exception - unsupported type
}

template <>
int convertFromDouble<int>(double value)
{
    return (int)value;
}

template <>
double convertFromDouble<double>(double value)
{
    return value;
}

template <>
float convertFromDouble<float>(double value)
{
    return (float)value;
}

template <>
long convertFromDouble<long>(double value)
{
    return (long)value;
}

//TODO: add time_point conversion support
