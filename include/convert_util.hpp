#include <chrono>
#include <vector>
#include "Eigen/Core"
using namespace std;

using vector_double = vector<double>;
using matrix_double = vector<vector_double>;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

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
matrix_double convertToMatrix(const vector<tuple<InputTypes...>> &input_tuple_matrix)
{
    matrix_double converted_matrix;
    for (auto it = input_tuple_matrix.begin(); it != input_tuple_matrix.end(); it++)
    {
        tuple<InputTypes...> input_tuple = *it;
        vector_double converted_vector = convertToVector<InputTypes...>(input_tuple);
        converted_matrix.push_back(converted_vector);
    }
    return converted_matrix;
}

template <Supported TargetType>
vector_double convertToVector(const vector<TargetType> &input_vector)
{
    vector_double converted_vector;
    for (auto it = input_vector.begin(); it != input_vector.end(); it++)
    {
        double converted_value = convertToDouble(*it);
        converted_vector.push_back(converted_value);
    }
    return converted_vector;
}

//TODO: hide helper methods
template <class Type>
void action(vector_double &vector, Type item)
{
    vector.push_back(convertToDouble(item));
}

template <size_t I, class... Types>
void fmap_impl(vector_double &vector, const tuple<Types...> &tuple)
{
    action(vector, std::get<I>(tuple));
    if constexpr (I + 1 < sizeof...(Types))
    {
        return fmap_impl<I + 1, Types...>(vector, tuple);
    }
}

template <typename... Types>
void fmap(vector_double &vector, const tuple<Types...> &tuple)
{
    return fmap_impl<0, Types...>(vector, tuple);
}

template <Supported... InputTypes>
vector_double convertToVector(const tuple<InputTypes...> &input_tuple)
{
    vector_double converted_vector;
    fmap(converted_vector, input_tuple);
    return converted_vector;
}

Matrix convertToEigen(const matrix_double &matrix)
{
    size_t columns = matrix.size();
    size_t rows = matrix[0].size(); //should be checked for every element
    Matrix m(rows, columns);
    for (size_t i = 0; i < columns; i++)
    {
        vector_double vector = matrix[i];
        for (size_t j = 0; j < rows; j++)
        {
            m(j, i) = vector[j];
        }
    }
    return m;
}

Vector convertToEigen(const vector_double &vector)
{
    size_t vector_size = vector.size();
    Vector v(vector_size);
    for (size_t i = 0; i < vector_size; i++)
    {
        v[i] = vector[i];
    }
    return v;
}

matrix_double convertFromEigen(const Matrix &matrix)
{
    size_t columns = matrix.cols();
    size_t rows = matrix.rows();
    matrix_double mat;
    for (size_t i = 0; i < columns; i++)
    {
        vector_double vec;
        for (size_t j = 0; j < rows; j++)
        {
            vec.push_back(matrix(j, i));
        }
        mat.push_back(vec);
    }
    return mat;
}

vector_double convertFromEigen(const Vector &vector)
{
    size_t vector_size = vector.size();
    vector_double v;
    for (size_t i = 0; i < vector_size; i++)
    {
        v.push_back(vector[i]);
    }
    return v;
}