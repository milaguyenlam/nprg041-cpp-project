#include <vector>
#include <tuple>
#include <string>
#include "convert_util.hpp"
using namespace std;

template <typename T>
concept Supported = is_same<T, int>::value || is_same<T, double>::value || is_same<T, float>::value || is_same<T, long>::value;

using matrix_double = vector<vector<double>>;
using vector_double = vector<double>;

template <Supported TargetType, Supported... InputTypes>
class Model
{
private:
    matrix_double convert(const vector<tuple<InputTypes...>> &input_matrix)
    {
        matrix_double converted_matrix;
        for (auto it = input_matrix.begin(); it != input_matrix.end(); it++)
        {
            tuple<InputTypes...> input_tuple = *it;
            vector_double converted_vector = convert(input_tuple);
            converted_matrix.push_back(converted_vector);
        }
        return converted_matrix;
    }

    vector_double convert(const vector<TargetType> &input_vector)
    {
        vector_double converted_vector;
        for (auto it = input_vector.begin(); it != input_vector.end(); it++)
        {
            double converted_value = convertToDouble(*it);
            converted_vector.push_back(converted_value);
        }
        return converted_vector;
    }

    vector_double convert(const tuple<InputTypes...> &input_vector)
    {
        vector_double converted_vector;
        tuple<InputTypes...> tuple = input_vector;
        size_t tuple_length = tuple_size<decltype(tuple)>::value;
        for (size_t i = 0; i < tuple_length; i++)
        {
            double converted_value = convertToDouble(get<0>(input_vector));
            converted_vector.push_back(converted_value);
        }
        return converted_vector;
    }

protected:
    vector<double> weights;
    virtual vector_double training_algorithm(const matrix_double &data, const vector_double &targets) = 0;

public:
    //overload for multiple target types instead of having a generic class??
    void fit(const vector<tuple<InputTypes...>> &training_data, const vector<TargetType> &training_targets)
    {
        matrix_double converted_data = convert(training_data);
        vector_double converted_targets = convert(training_targets);
        training_algorithm(converted_data, converted_targets);
    }

    TargetType predict(const tuple<InputTypes...> &predict_vector)
    {
        vector_double converted_vector = convert(predict_vector);
        //convert tuple into vector<double> and compute dot product of given vector and weights
        //convert result into TargetType and return
    }

    vector<TargetType> predict(const vector<tuple<InputTypes...>> &predict_vectors)
    {
        //convert tuples into vector<double> and compute dot product of a given vector and weights
        //convert result into TargetType and return
    }
};

template <Supported TargetType, Supported... InputTypes>
class LinearRegression : public Model<TargetType, InputTypes...>
{
protected:
    vector_double training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};

template <Supported TargetType, Supported... InputTypes>
class LogisticRegression : public Model<TargetType, InputTypes...>
{
protected:
    vector_double training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};

template <Supported TargetType, Supported... InputTypes>
class SVM : public Model<TargetType, InputTypes...>
{
protected:
    vector_double training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};
