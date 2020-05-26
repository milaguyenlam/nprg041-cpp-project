#include <vector>
#include <tuple>
#include <string>
#include "convert_util.hpp"
using namespace std;

// template <typename T>
// concept Supported = is_same<T, int>::value || is_same<T, double>::value || is_same<T, float>::value || is_same<T, long>::value;

using vector_double = vector<double>;
using matrix_double = vector<vector_double>;

//TODO: import math library

template <Supported TargetType, Supported... InputTypes>
class Model
{
protected:
    vector<double> weights;
    virtual void training_algorithm(const matrix_double &data, const vector_double &targets) = 0;

public:
    //overload for multiple target types instead of having a generic class??
    void fit(const vector<tuple<InputTypes...>> &training_data, const vector<TargetType> &training_targets)
    {
        matrix_double converted_data = convertToMatrix<InputTypes...>(training_data);
        vector_double converted_targets = convertToVector<TargetType>(training_targets);
        training_algorithm(converted_data, converted_targets);
    }

    //TODO: implement predict() methods
    TargetType predict(const tuple<InputTypes...> &predict_vector) const
    {
        vector_double converted_vector = convertToVector<InputTypes...>(predict_vector);
        //convert tuple into vector<double> and compute dot product of given vector and weights
        //convert result into TargetType and return
    }

    vector<TargetType> predict(const vector<tuple<InputTypes...>> &predict_vectors) const
    {
        matrix_double converted_vectors = convertToMatrix<InputTypes...>(predict_vectors);
        //convert tuples into vector<double> and compute dot product of a given vector and weights
        //convert result into TargetType and return
    }
};

//TODO: implement Linear Regression algorithm
template <Supported TargetType, Supported... InputTypes>
class LinearRegression : public Model<TargetType, InputTypes...>
{
protected:
    void training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};

//TODO: implement Logistic Regression algorithm
template <Supported TargetType, Supported... InputTypes>
class LogisticRegression : public Model<TargetType, InputTypes...>
{
protected:
    void training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};

//TODO: implement SVM algorithm
template <Supported TargetType, Supported... InputTypes>
class SVM : public Model<TargetType, InputTypes...>
{
protected:
    void training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
    }
};
