#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include "convert_util.hpp"

using Vector1d = Eigen::Matrix<double, 1, 1>;

//TODO: add include guards
//TODO: consider deduction guides

class Model
{
protected:
    Eigen::VectorXd weights;
    virtual void training_algorithm(const Eigen::MatrixXd &data, const Eigen::VectorXd &targets) = 0;

public:
    virtual ~Model()
    {
    }
    void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &training_targets)
    {
        training_algorithm(training_data, training_targets);
    }

    Vector1d predict(const Eigen::VectorXd &predict_vector) const
    {
        return predict_vector.transpose() * weights;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &predict_vectors) const
    {
        return predict_vectors.transpose() * weights;
    }
};

template <Supported TargetType, Supported... TupleTypes>
class ModelConvertProxy
{
private:
    Model &model;

public:
    ModelConvertProxy(Model &model_to_wrap) : model(model_to_wrap)
    {
    }

    void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets)
    {
        auto converted_data = convert_from_std<TupleTypes...>(data);
        auto converted_targets = convert_from_std<TargetType>(targets);
        model.fit(converted_data, converted_targets);
    }

    TargetType predict(const std::tuple<TupleTypes...> &tuple)
    {
        auto converted_vector = convert_from_std<TupleTypes...>(tuple);
        Vector1d value = model.predict(converted_vector);
        TargetType ret_value = convert_to_std<TargetType>(value[0]);
        return ret_value;
    }

    std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples)
    {
        auto converted_matrix = convert_from_std<TupleTypes...>(tuples);
        auto value_vector = model.predict(converted_matrix);
        std::vector<TargetType> ret_values = convert_to_std<TargetType>(value_vector);
        return ret_values;
    }
};

//TODO: implement method chaining?
class LinearRegression : public Model
{
private:
    double lambda = 0;
    std::size_t iterations = 100;
    double learning_rate = 0.001;

protected:
    void training_algorithm(const Eigen::MatrixXd &data, const Eigen::VectorXd &targets) override
    {
        std::size_t input_vector_dimension = data.rows();
        std::size_t input_data_length = data.cols();
        weights = Eigen::VectorXd(input_vector_dimension).setZero();

        for (std::size_t iter = 0; iter < iterations; iter++)
        {
            for (std::size_t i = 0; i < input_data_length; i++)
            {
                Eigen::VectorXd input_vector = data.col(i);
                double target = targets(i);
                auto prediction = input_vector.transpose() * weights;
                auto regularization = lambda * weights;
                auto error = prediction - target;
                auto step = learning_rate * (error * input_vector) + regularization;
                weights = weights - step;
            }
        }
    }

public:
    ~LinearRegression() override
    {
    }
    LinearRegression &with_L2_regulatization(double new_lambda)
    {
        lambda = new_lambda;
        return *this;
    }
    LinearRegression &set_iterations(std::size_t new_iterations)
    {
        iterations = new_iterations;
        return *this;
    }
    LinearRegression &set_learning_rate(double alpha)
    {
        learning_rate = alpha;
        return *this;
    }
};

//TODO: implement SVM and Logistic Regression classes