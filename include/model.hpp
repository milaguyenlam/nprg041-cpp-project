#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "convert_util.hpp"
#include "model_exceptions.hpp"

using Vector1d = Eigen::Matrix<double, 1, 1>;
using Vector1i = Eigen::Matrix<int, 1, 1>;

//TODO: add include guards
//TODO: consider deduction guides for proxy classes

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
    virtual Vector1i make_prediction(const Eigen::VectorXd &vector) const = 0;
    virtual Eigen::VectorXi make_prediction(const Eigen::MatrixXd &vectors) const = 0;
    bool apply_padding = true;
    bool binary_classification = false;

public:
    ~ClassificationModel() override
    {
    }
    void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXi &training_targets)
    {
        if (training_data.cols() != training_targets.size())
        {
            throw InvalidTrainingDataFormat();
        }
        training_data_dimension = training_data.rows();
        if (apply_padding)
        {
            auto padded_data = pad_matrix_with_ones(training_data);
            training_algorithm(padded_data, training_targets);
        }
        else
        {
            training_algorithm(training_data, training_targets);
        }
        trained = true;
    }
    Vector1i predict(const Eigen::VectorXd &vector) const
    {
        if (training_data_dimension != vector.size())
        {
            throw InvalidPredictDataFormat();
        }
        if (!trained)
        {
            throw PredictBeforeFit();
        }
        if (apply_padding)
        {
            auto padded_vector = pad_vector_with_ones(vector);
            return make_prediction(padded_vector);
        }
        else
        {
            return make_prediction(vector);
        }
    }
    Eigen::VectorXi predict(const Eigen::MatrixXd &vectors) const
    {
        if (training_data_dimension != vectors.rows())
        {
            throw InvalidPredictDataFormat();
        }
        if (!trained)
        {
            throw PredictBeforeFit();
        }
        if (apply_padding)
        {
            auto padded_vectors = pad_matrix_with_ones(vectors);
            return make_prediction(padded_vectors);
        }
        else
        {
            return make_prediction(vectors);
        }
    }

private:
    bool trained = false;
    long int training_data_dimension;
    void check_targets(const Eigen::VectorXi &targets)
    {
        std::set<int> found_target_labels;
        std::size_t length = targets.size();
        for (std::size_t i = 0; i < length; i++)
        {
            found_target_labels.insert(targets[i]);
        }
        int elements_count = found_target_labels.size();
        for (int i = 0; i < elements_count; i++)
        {
            if (!found_target_labels.contains(i))
            {
                throw InvalidTargetFormat();
            }
        }
        if ((elements_count != 2) && binary_classification)
        {
            throw BinaryLabelsExpected();
        }
    }
};

class RegressionModel : public Model
{
protected:
    virtual void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXd &targets) = 0;
    virtual Vector1d make_prediction(const Eigen::VectorXd &vector) const = 0;
    virtual Eigen::VectorXd make_prediction(const Eigen::MatrixXd &vectors) const = 0;
    bool apply_padding = true;

public:
    virtual ~RegressionModel()
    {
    }
    void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &training_targets)
    {
        if (training_data.cols() != training_targets.size())
        {
            throw InvalidTrainingDataFormat();
        }
        training_data_dimension = training_data.rows();
        if (apply_padding)
        {
            auto padded_data = pad_matrix_with_ones(training_data);
            training_algorithm(padded_data, training_targets);
        }
        else
        {
            training_algorithm(training_data, training_targets);
        }
        trained = true;
    }
    Vector1d predict(const Eigen::VectorXd &vector) const
    {
        if (training_data_dimension != vector.size())
        {
            throw InvalidPredictDataFormat();
        }
        if (!trained)
        {
            throw PredictBeforeFit();
        }
        if (apply_padding)
        {
            auto padded_vector = pad_vector_with_ones(vector);
            return make_prediction(padded_vector);
        }
        else
        {
            return make_prediction(vector);
        }
    }
    Eigen::VectorXd predict(const Eigen::MatrixXd &vectors) const
    {
        if (training_data_dimension != vectors.rows())
        {
            throw InvalidPredictDataFormat();
        }
        if (!trained)
        {
            throw PredictBeforeFit();
        }
        if (apply_padding)
        {
            auto padded_vectors = pad_matrix_with_ones(vectors);
            return make_prediction(padded_vectors);
        }
        else
        {
            return make_prediction(vectors);
        }
    }

private:
    bool trained = false;
    long int training_data_dimension;
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
        std::size_t data_dimension = data.rows();
        std::size_t data_length = data.cols();
        weights = Eigen::MatrixXd(data_dimension, 1).setZero();

        for (std::size_t iter = 0; iter < iterations; iter++)
        {
            for (std::size_t i = 0; i < data_length; i++)
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
private:
    double learning_rate = 0.01;
    std::size_t iterations = 1000;
    bool dataset_separated = false;

protected:
    void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &targets) override
    {
        std::size_t data_lenght = dataset.cols();
        std::size_t data_dimension = dataset.rows();
        weights = Eigen::MatrixXd(data_dimension, 1).setZero();

        for (size_t it = 0; it < iterations; it++)
        {
            bool incorrect_classification = false;
            for (size_t i = 0; i < data_lenght; i++)
            {
                Eigen::VectorXd vector = dataset.col(i);
                auto target = targets[i];
                auto prediction = make_prediction(vector)(0, 0);
                if (prediction != target)
                {
                    incorrect_classification = true;
                    auto error = target - prediction;
                    weights = weights + (learning_rate * error) * vector;
                }
            }
            if (!incorrect_classification)
            {
                dataset_separated = true;
                return;
            }
        }
    };
    Vector1i make_prediction(const Eigen::VectorXd &vector) const override
    {
        auto dot = (vector.transpose() * weights)(0, 0);
        Vector1i result;
        if (dot > 0)
        {
            result[0] = 1;
        }
        else
        {
            result[0] = 0;
        }
        return result;
    };
    Eigen::VectorXi make_prediction(const Eigen::MatrixXd &vectors) const override
    {
        std::size_t length = vectors.cols();
        Eigen::VectorXi ret_vector(length);
        Eigen::VectorXd dot = vectors.transpose() * weights;
        for (size_t i = 0; i < length; i++)
        {
            if (dot[i] > 0)
            {
                ret_vector[i] = 1;
            }
            else
            {
                ret_vector[i] = 0;
            }
        }
        return ret_vector;
    };

public:
    Perceptron()
    {
        binary_classification = true;
    }
    ~Perceptron() override {}
    bool was_dataset_separated()
    {
        return dataset_separated;
    }
    Perceptron &set_iterations(std::size_t new_iterations)
    {
        iterations = new_iterations;
        return *this;
    }
};

class BinarySVM : public ClassificationModel
{
protected:
    void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &input_targets) override
    {
        Eigen::VectorXi targets = change_target_values(input_targets);
        train_targets = targets;
        train_data = dataset.replicate(1, 1);
        std::size_t dataset_size = dataset.cols();
        weights = Eigen::VectorXd(dataset_size).setZero();
        b = 0;
        compute_kernel_values(dataset);
        std::size_t passes = 0;
        while (passes < max_passes)
        {
            std::size_t a_changed = 0;
            std::size_t a_length = weights.rows();
            for (std::size_t i = 0; i < a_length; i++)
            {

                double error = predict_value(dataset.col(i)) - (double)targets[i];
                if (((weights(i, 0) < regularization_param) && (targets[i] * error < -numerical_tolerance)) || ((weights(i, 0) > 0) && (targets[i] * error > numerical_tolerance)))
                {
                    std::size_t j = std::rand() % (a_length - 1);
                    while (j == i)
                    {
                        j = std::rand() % (a_length - 1);
                    }
                    double second = 2 * kernel_values(i, j) - kernel_values(i, i) - kernel_values(j, j);
                    if (second >= -numerical_tolerance)
                    {
                        continue;
                    }
                    double error_j = predict_value(dataset.col(j)) - targets[j];
                    double weight_j = weights(j, 0) - (targets[j] * (error - error_j)) / second;

                    double L;
                    double H;

                    if (targets[i] == targets[j])
                    {
                        L = std::max(0.0, weights(i, 0) + weights(j, 0) - regularization_param);
                        H = std::min(regularization_param, weights(i, 0) + weights(j, 0));
                    }
                    else
                    {
                        L = std::max(0.0, weights(j, 0) - weights(i, 0));
                        H = std::min(regularization_param, regularization_param + weights(j, 0) - weights(i, 0));
                    }

                    std::clamp(weight_j, L, H);

                    if ((H - L < numerical_tolerance) || (std::abs(weight_j - weights(j, 0)) < numerical_tolerance))
                    {
                        continue;
                    }

                    double weight_i = weights(i, 0) - (targets[i] * targets[j] * (weight_j - weights(j, 0)));
                    double b_i = b - error - (targets[i] * (weight_i - weights(i, 0)) * kernel_values(i, i)) - (targets[j] * (weight_j - weights(j, 0)) * kernel_values(j, i));
                    double b_j = b - error_j - (targets[i] * (weight_i - weights(i, 0)) * kernel_values(i, j)) - (targets[j] * (weight_j - weights(j, 0)) * kernel_values(j, j));

                    if ((0.0 < weight_i) && (weight_i < regularization_param))
                    {
                        b = b_i;
                    }
                    else if ((0.0 < weight_j) && (weight_j < regularization_param))
                    {
                        b = b_j;
                    }
                    else
                    {
                        b = (b_i + b_j) / 2.0;
                    }

                    weights(i, 0) = weight_i;
                    weights(j, 0) = weight_j;

                    a_changed++;
                }
            }
            if (a_changed > 0)
            {
                passes = 0;
            }
            else
            {
                passes++;
            }
        }
    }
    Vector1i make_prediction(const Eigen::VectorXd &vector) const override
    {
        Vector1i value;
        auto prediction = predict_value(vector);
        if (prediction > 0)
        {
            value[0] = 1;
        }
        else
        {
            value[0] = 0;
        }
        return value;
    }
    Eigen::VectorXi make_prediction(const Eigen::MatrixXd &vectors) const override
    {
        std::size_t length = vectors.cols();
        Eigen::VectorXi return_vector(length);
        for (size_t i = 0; i < length; i++)
        {
            Eigen::VectorXd vector = vectors.col(i);
            return_vector[i] = make_prediction(vector)(0, 0);
        }
        return return_vector;
    }

private:
    double regularization_param = 1;
    double numerical_tolerance = 0.0001;
    double b = 0;
    std::size_t max_passes = 5;
    Eigen::MatrixXd kernel_values;
    Eigen::VectorXi train_targets;
    Eigen::MatrixXd train_data;

    void compute_kernel_values(const Eigen::MatrixXd &dataset)
    {
        std::size_t data_size = dataset.cols();
        Eigen::MatrixXd kernel_vals(data_size, data_size);
        for (size_t i = 0; i < data_size; i++)
        {
            for (size_t j = 0; j < data_size; j++)
            {
                Eigen::VectorXd vector_i = dataset.col(i);
                Eigen::VectorXd vector_j = dataset.col(j);
                kernel_vals(i, j) = compute_kernel(vector_i, vector_j);
            }
        }
        kernel_values = kernel_vals;
    }

    double compute_kernel(const Eigen::VectorXd &vector, const Eigen::VectorXd &vector2) const
    {
        return (vector.transpose() * vector2)(0.0);
    }

    double predict_value(const Eigen::VectorXd &vector) const
    {
        double sum = 0;
        std::size_t weights_size = weights.rows();
        for (size_t i = 0; i < weights_size; i++)
        {
            Eigen::VectorXd train_vector_i = train_data.col(i);
            sum += weights(i, 0) * train_targets[i] * compute_kernel(train_vector_i, vector);
        }
        return sum + b;
    }

    Eigen::VectorXi change_target_values(const Eigen::VectorXi &input_targets)
    {
        std::size_t length = input_targets.size();
        Eigen::VectorXi targets(length);
        for (size_t i = 0; i < length; i++)
        {
            if (input_targets[i] == 0)
            {
                targets[i] = -1;
            }
            else
            {
                targets[i] = 1;
            }
        }
        return targets;
    }

public:
    BinarySVM()
    {
        apply_padding = false;
    }
    ~BinarySVM()
    {
    }
    BinarySVM &set_regularization_param(double reg_param)
    {
        regularization_param = reg_param;
        return *this;
    }
    BinarySVM &set_tolerance(double tolerance)
    {
        numerical_tolerance = tolerance;
        return *this;
    }
    BinarySVM &set_max_passes(std::size_t passes)
    {
        max_passes = passes;
        return *this;
    }
};

// Convert Proxy Classes

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
        auto ret_values = VectorConverter::convert_to_std(values, label_map);
        return ret_values;
    }
};
