#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include "convert_util.hpp"

using namespace std;
using vector_double = vector<double>;
using matrix_double = vector<vector_double>;

//TODO: rewrite Model class to support only Eigen data structures
//TODO: add virtual destructor to Model class
//TODO: add include guards

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
private:
    double lambda = 0;
    int iterations = 100;
    double learning_rate = 0.001;

protected:
    void training_algorithm(const matrix_double &data, const vector_double &targets) override
    {
        Vector etargets = convertToEigen(targets);
        Matrix edata = convertToEigen(data);
        size_t data_length = targets.size();
        size_t weights_lenght = edata.rows();
        Vector eweights(weights_lenght);
        eweights.setZero();

        for (int i = 0; i < iterations; i++)
        {
            for (size_t j = 0; j < data_length; j++)
            {
                auto evector = edata.col(j);
                auto etarget = etargets.row(j);
                auto res = evector.transpose() * eweights - etarget;
                auto res2 = lambda * eweights;
                auto res3 = res * evector.transpose();
                auto res4 = learning_rate * (res3.transpose() + res2);
                auto res5 = eweights - res4;
                eweights = res5;
                //cout << eweights << endl;
                //eweights = eweights - learning_rate * ((evector.transpose() * eweights - etarget) * evector + lambda * eweights);
            }
        }
        cout << eweights << endl;
    }

public:
    LinearRegression with_regulatization(double lambda)
    {
        this.lambda = lambda;
        return this;
    }
    LinearRegression set_iterations(int iterations)
    {
        this.iterations = iterations;
        return this;
    }
    LinearRegression set_learning_rate(double alpha)
    {
        learning_rate = alpha;
        return this;
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