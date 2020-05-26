#include <chrono>
#include <vector>
using namespace std;

using Vector = vector<double>;
using Matrix = vector<Vector>;

template <typename T>
concept Supported = is_same<T, int>::value || is_same<T, double>::value || is_same<T, float>::value || is_same<T, long>::value;

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

template <Supported T>
T convertFromDouble(double value)
{
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
//TODO: refactor complex convert methods

template <Supported... InputTypes>
Matrix convertToMatrix(const vector<tuple<InputTypes...>> &input_tuple_matrix)
{
    Matrix converted_matrix;
    for (auto it = input_tuple_matrix.begin(); it != input_tuple_matrix.end(); it++)
    {
        tuple<InputTypes...> input_tuple = *it;
        Vector converted_vector = convertToVector<InputTypes...>(input_tuple);
        converted_matrix.push_back(converted_vector);
    }
    return converted_matrix;
}

template <Supported TargetType>
Vector convertToVector(const vector<TargetType> &input_vector)
{
    Vector converted_vector;
    for (auto it = input_vector.begin(); it != input_vector.end(); it++)
    {
        double converted_value = convertToDouble(*it);
        converted_vector.push_back(converted_value);
    }
    return converted_vector;
}

//TODO: hide helper methods
template <class Type>
void action(Vector &vector, Type item)
{
    vector.push_back(convertToDouble(item));
}

template <size_t I, class... Types>
void fmap_impl(Vector &vector, const tuple<Types...> &tuple)
{
    action(vector, std::get<I>(tuple));
    if constexpr (I + 1 < sizeof...(Types))
    {
        return fmap_impl<I + 1, Types...>(vector, tuple);
    }
}

template <typename... Types>
void fmap(Vector &vector, const tuple<Types...> &tuple)
{
    return fmap_impl<0, Types...>(vector, tuple);
}

template <Supported... InputTypes>
Vector convertToVector(const tuple<InputTypes...> &input_tuple)
{
    Vector converted_vector;
    fmap(converted_vector, input_tuple);
    return converted_vector;
}