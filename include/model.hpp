#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include "convert_util.hpp"

using Vector1d = Eigen::Matrix<double, 1, 1>;
using Vector1i = Eigen::Matrix<int, 1, 1>;

//TODO: add include guards
//TODO: consider deduction guides
//TODO: add initial padding when fitting

class Model
{
protected:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd pad_matrix_with_ones(const Eigen::MatrixXd &dataset) const
    {
        std::size_t rows = dataset.rows();
        std::size_t cols = dataset.cols();
        Eigen::MatrixXd padded_matrix(rows + 1, cols);
        for (size_t i = 0; i < rows + 1; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                if (i == rows)
                {
                    padded_matrix(i, j) = 1.0;
                }
                else
                {
                    padded_matrix(i, j) = dataset(i, j);
                }
            }
        }
        return padded_matrix;
    }
    Eigen::VectorXd pad_vector_with_ones(const Eigen::VectorXd &vector) const
    {
        std::size_t size = vector.size();
        Eigen::VectorXd padded_vector(size + 1);
        for (size_t i = 0; i < size + 1; i++)
        {
            if (i == size)
            {
                padded_vector[i] = 1.0;
            }
            else
            {
                padded_vector[i] = vector[i];
            }
        }
        return padded_vector;
    }

public:
    virtual ~Model()
    {
    }
};

class ClassificationModel : public Model
{
protected:
    virtual void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &targets) = 0;
    std::map<int, int> label_map;

public:
    ~ClassificationModel() override
    {
    }
    void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXi &training_targets)
    {
        auto padded_data = pad_matrix_with_ones(training_data);
        //TODO: map targets to inner targets
        training_algorithm(padded_data, training_targets);
    }
    virtual Vector1i predict(const Eigen::VectorXd &vector) const = 0;
    virtual Eigen::VectorXi predict(const Eigen::MatrixXd &vectors) const = 0;
};

class RegressionModel : public Model
{
protected:
    virtual void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXd &targets) = 0;
    virtual Vector1d make_prediction(const Eigen::VectorXd &vector) const = 0;
    virtual Eigen::VectorXd make_prediction(const Eigen::MatrixXd &vectors) const = 0;

public:
    virtual ~RegressionModel()
    {
    }
    void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &training_targets)
    {
        auto padded_data = pad_matrix_with_ones(training_data);
        training_algorithm(padded_data, training_targets);
    }
    Vector1d predict(const Eigen::VectorXd &vector) const
    {
        auto padded_vector = pad_vector_with_ones(vector);
        return make_prediction(padded_vector);
    }
    Eigen::VectorXd predict(const Eigen::MatrixXd &vectors) const
    {
        auto padded_vectors = pad_matrix_with_ones(vectors);
        return make_prediction(padded_vectors);
    }
};

class LinearRegression : public RegressionModel
{
private:
    double lambda = 0;
    std::size_t iterations = 100;
    double learning_rate = 0.01;

protected:
    void
    training_algorithm(const Eigen::MatrixXd &data, const Eigen::VectorXd &targets) override
    {
        std::size_t input_vector_dimension = data.rows();
        std::size_t input_data_length = data.cols();
        weights = Eigen::MatrixXd(input_vector_dimension, 1).setZero();

        for (std::size_t iter = 0; iter < iterations; iter++)
        {
            for (std::size_t i = 0; i < input_data_length; i++)
            {
                Eigen::VectorXd input_vector = data.col(i);
                double target = targets(i);
                auto prediction = (input_vector.transpose() * weights)(0, 0);
                auto regularization = lambda * weights;
                auto error = prediction - target;
                auto step = learning_rate * (error * input_vector) + regularization;
                weights = weights - step;
            }
        }
    }
    Vector1d make_prediction(const Eigen::VectorXd &vector) const override
    {

        return vector.transpose() * weights;
    }
    Eigen::VectorXd make_prediction(const Eigen::MatrixXd &vectors) const override
    {

        return vectors.transpose() * weights;
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

class Perceptron : public ClassificationModel
{
public:
    ~Perceptron() override
    {
    }
};

//TODO: implement SVM and Logistic Regression classes

// Convert Proxy Classes
//TODO: implement Vector and Array conversions if possible
template <Supported TargetType, Supported... TupleTypes>
class RegressionModelProxy
{
private:
    RegressionModel &model;

public:
    RegressionModelProxy(RegressionModel &model_to_wrap) : model(model_to_wrap)
    {
    }

    void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets)
    {
        auto converted_data = convert_from_std<TupleTypes...>(data);
        auto converted_targets = VectorConverter::convert_from_std<TargetType>(targets);
        model.fit(converted_data, converted_targets);
    }

    TargetType predict(const std::tuple<TupleTypes...> &tuple)
    {
        auto converted_vector = convert_from_std<TupleTypes...>(tuple);
        Vector1d value = model.predict(converted_vector);
        TargetType ret_value = SingleValueConverter::convert_to<TargetType>(value[0]);
        return ret_value;
    }

    std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples)
    {
        auto converted_matrix = convert_from_std<TupleTypes...>(tuples);
        auto value_vector = model.predict(converted_matrix);
        std::vector<TargetType> ret_values = VectorConverter::convert_to_std<TargetType>(value_vector);
        return ret_values;
    }
};

//TODO: create concept for classification target type
template <typename TargetType, Supported... TupleTypes>
class ClassificationModelProxy
{
private:
    ClassificationModel &model;
    std::map<int, TargetType> label_map;

public:
    ClassificationModelProxy(ClassificationModel &model_to_wrap) : model(model_to_wrap)
    {
    }
    void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets)
    {
        auto converted_data = convert_from_std(data);
        auto converted_targets = VectorConverter::convert_from_std(targets, label_map);
        model.fit(converted_data, converted_targets);
    }
    TargetType predict(const std::tuple<TupleTypes...> &tuple)
    {
        auto converted_vector = convert_from_std<TupleTypes...>(tuple);
        auto value = model.predict(converted_vector);
        TargetType ret_value = label_map.find(value[0])->second;
        return ret_value;
    }
    std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples)
    {
        auto converted_matrix = convert_from_std<TupleTypes...>(tuples);
        auto values = model.predict(converted_matrix);
        auto ret_values = convert_to_std(values, label_map);
        return ret_values;
    }
};
