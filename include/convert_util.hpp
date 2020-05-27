#include <chrono>
#include <vector>
#include "Eigen/Core"

//TODO: remove using's
//TODO: refactor whole convert_utils.hpp file with usability in mind

using namespace std;

template <typename T>
concept Supported = is_same<T, int>::value || is_same<T, double>::value || is_same<T, float>::value || is_same<T, long>::value;

template <class Type>
void push_value_to_vector(std::vector<double> &vector, Type item)
{
    vector.push_back(convertToDouble(item));
}

template <size_t I, class... Types>
void iterate_recursively_and_push_value(std::vector<double> &vector, const tuple<Types...> &tuple)
{
    push_value_to_vector(vector, std::get<I>(tuple));
    if constexpr (I + 1 < sizeof...(Types))
    {
        return iterate_recursively_and_push_value<I + 1, Types...>(vector, tuple);
    }
}

template <typename... Types>
void push_converted_data_to_vector(std::vector<double> &vector, const tuple<Types...> &tuple)
{
    return iterate_recursively_and_push_value<0, Types...>(vector, tuple);
}

// double convertToDouble(int value)
// {
//     return (double)value;
// };

// double convertToDouble(double value)
// {
//     return value;
// };

// double convertToDouble(long value)
// {
//     return (double)value;
// };

// template <Supported T>
// T convertFromDouble(double value)
// {
// }

// template <>
// int convertFromDouble<int>(double value)
// {
//     return (int)value;
// }

// template <>
// double convertFromDouble<double>(double value)
// {
//     return value;
// }

// template <>
// float convertFromDouble<float>(double value)
// {
//     return (float)value;
// }

// template <>
// long convertFromDouble<long>(double value)
// {
//     return (long)value;
// }

//TODO: add time_point conversion support

// template <Supported... InputTypes>
// Matrix convert_to_eigen(const vector<tuple<InputTypes...>> &input_tuple_matrix)
// {
//     std::vector<std::vector<double>> matrix = convert_to_vector<InputTypes...>(input_tuple_matrix);
//     Matrix result_matrix = convert_to_eigen(matrix);
//     return result_matrix;
// }

// template <Supported... InputTypes>
// Matrix convert_to_eigen(const vector<tuple<InputTypes...>> &input_tuple_matrix)
// {
//     std::vector<std::vector<double>> matrix = convert_to_vector<InputTypes...>(input_tuple_matrix);
//     Matrix result_matrix = convert_to_eigen(matrix);
//     return result_matrix;
// }

// template <Supported... InputTypes>
// std::vector<std::vector<double>> convert_to_vector(const vector<tuple<InputTypes...>> &input_tuple_matrix)
// {
//     std::vector<std::vector<double>> converted_matrix;
//     for (auto it = input_tuple_matrix.begin(); it != input_tuple_matrix.end(); it++)
//     {
//         tuple<InputTypes...> input_tuple = *it;
//         std::vector<double> converted_vector = convertToVector<InputTypes...>(input_tuple);
//         converted_matrix.push_back(converted_vector);
//     }
//     return converted_matrix;
// }

// template <Supported TargetType>
// std::vector<double> convertToVector(const vector<TargetType> &input_vector)
// {
//     std::vector<double> converted_vector;
//     for (auto it = input_vector.begin(); it != input_vector.end(); it++)
//     {
//         double converted_value = convertToDouble(*it);
//         converted_vector.push_back(converted_value);
//     }
//     return converted_vector;
// }

// //TODO: hide helper methods

// template <Supported... InputTypes>
// std::vector<double> convertToVector(const tuple<InputTypes...> &input_tuple)
// {
//     std::vector<double> converted_vector;
//     push_converted_data_to_vector(converted_vector, input_tuple);
//     return converted_vector;
// }

// Matrix convert_to_eigen(const std::vector<std::vector<double>> &matrix)
// {
//     size_t columns = matrix.size();
//     size_t rows = matrix[0].size(); //should be checked for every element
//     Matrix m(rows, columns);
//     for (size_t i = 0; i < columns; i++)
//     {
//         std::vector<double> vector = matrix[i];
//         for (size_t j = 0; j < rows; j++)
//         {
//             m(j, i) = vector[j];
//         }
//     }
//     return m;
// }

// Vector convert_to_eigen(const std::vector<double> &vector)
// {
//     size_t vector_size = vector.size();
//     Vector v(vector_size);
//     for (size_t i = 0; i < vector_size; i++)
//     {
//         v[i] = vector[i];
//     }
//     return v;
// }

// std::vector<std::vector<double>> convert_from_eigen(const Matrix &matrix)
// {
//     size_t columns = matrix.cols();
//     size_t rows = matrix.rows();
//     std::vector<std::vector<double>> mat;
//     for (size_t i = 0; i < columns; i++)
//     {
//         std::vector<double> vec;
//         for (size_t j = 0; j < rows; j++)
//         {
//             vec.push_back(matrix(j, i));
//         }
//         mat.push_back(vec);
//     }
//     return mat;
// }

// std::vector<double> convert_from_eigen(const Vector &vector)
// {
//     size_t vector_size = vector.size();
//     std::vector<double> v;
//     for (size_t i = 0; i < vector_size; i++)
//     {
//         v.push_back(vector[i]);
//     }
//     return v;
// }

//public API methods

//is overriden(time_point case), should be called only for long, double, int, float
template <Supported From>
double convert_from(From value)
{
    return (double)value;
}

//TODO: ask if convert_from/to<time_point>() implementation is correct
template <class Clock, class Duration>
double convert_from(const std::chrono::time_point<Clock, Duration> &value)
{
    return (double)(value.time_since_epoch().count());
}

//is overriden, should never be called
template <Supported To>
To convert_to(double value)
{
}

template <>
double convert_to(double value)
{
    return value;
}

template <>
int convert_to(double value)
{
    return (int)value;
}

template <>
float convert_to(double value)
{
    return (float)value;
}

template <>
long convert_to(double value)
{
    return (long)value;
}

template <class Clock, class Duration>
std::chrono::time_point<Clock, Duration> convert_to(double value)
{
}

template <Supported From>
Eigen::VectorXd convert_from(const std::vector<From> &vector)
{
    size_t vector_length = vector.size();
    Eigen::VectorXd evector(vector_length);
    for (size_t i = 0; i < vector_length; i++)
    {
        evector[i] = vector[i];
    }
    return evector;
}

template <Supported To>
std::vector<To> convert_to(const Eigen::VectorXd &vector)
{
    std::vector<To> svector;
    size_t vector_lenght = vector.rows();
    for (size_t i = 0; i < vector_lenght; i++)
    {
        To value = convert_to<To>(vector[i]);
        svector.push_back(value);
    }
    return svector;
}

template <Supported... From>
Eigen::VectorXd convert_from(const std::tuple<From...> &tuple)
{
    std::vector<double> vector;
    push_converted_data_to_vector(vector, tuple);
    Eigen::VectorXd evector = convert_from<double>(vector);
    return evector;
}

//unnecessary??
template <Supported... To>
std::tuple<To...> convert_to(const Eigen::VectorXd &vector)
{
}

template <Supported From>
Eigen::MatrixXd convert_from(const std::vector<std::vector<From>> &matrix)
{
    size_t columns = matrix.size();
    size_t rows = matrix[0].size(); //should be checked for every element
    Eigen::MatrixXd m(rows, columns);
    for (size_t i = 0; i < columns; i++)
    {
        auto vector = matrix[i];
        for (size_t j = 0; j < rows; j++)
        {
            m(j, i) = vector[j];
        }
    }
    return m;
}

template <Supported To>
std::vector<std::vector<To>> convert_to(const Eigen::MatrixXd &matrix)
{
}

template <Supported... From>
Eigen::MatrixXd convert_from(const std::vector<std::tuple<From...>> &tuples)
{
    std::vector<std::vector<double>> matrix;
    std::size_t tuples_size = tuples.size();
    for (size_t i = 0; i < tuples_size; i++)
    {
        std::vector<double> vector = convert_to<double>(tuples[i]);
        matrix.push_back(vector);
    }
    Eigen::MatrixXd ret_matrix = convert_from<double>(matrix);
    return ret_matrix;
}

template <Supported... To>
std::vector<std::tuple<To...>> convert_to(const Eigen::MatrixXd &vector)
{
}