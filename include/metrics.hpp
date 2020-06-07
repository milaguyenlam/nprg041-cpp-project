#ifndef MLLIB_METRICS_HPP
#define MLLIB_METRICS_HPP

#include "Eigen/Core"
#include "convert_util.hpp"
#include "metrics_exceptions.hpp"

namespace metrics
{
    using namespace convert;

    //method computing accuracy from Eigen vectors of int, one with predicted values, the other with actual targets
    //accuracy = correctly_predicted / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    double compute_classification_accuracy(Eigen::VectorXi &predicted, Eigen::VectorXi &actual)
    {
        std::size_t predicted_size = predicted.size();
        std::size_t actual_size = actual.size();
        if (predicted_size != actual_size)
        {
            throw InvalidDataFormat();
        }
        std::size_t correct_counter = 0;
        for (size_t i = 0; i < predicted_size; i++)
        {
            if (predicted[i] == actual[i])
            {
                correct_counter++;
            }
        }

        return (double)correct_counter / (double)predicted_size;
    }

    //method for computing mean squared error from Eigen vectors of double, one with predicted values, the other with actual targets
    //mse = sum((prediction - actual_value)^2) / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    double compute_mse(Eigen::VectorXd &predicted, Eigen::VectorXd &actual)
    {
        std::size_t predicted_size = predicted.size();
        std::size_t actual_size = actual.size();
        if (predicted_size != actual_size)
        {
            throw InvalidDataFormat();
        }
        double sum = 0;
        for (size_t i = 0; i < predicted_size; i++)
        {
            double squared_error = std::pow(predicted[i] - actual[i], 2);
            sum += squared_error;
        }
        return sum / (double)predicted_size;
    }

    //wrapper for std type vector representations
    //method computing accuracy from Eigen vectors of int, one with predicted values, the other with actual targets
    //accuracy = correctly_predicted / number_of_predictions
    template <SupportedClassificationTargets TargetType>
    double compute_classification_accuracy(std::vector<TargetType> &predicted, std::vector<TargetType> &actual)
    {
        std::map<int, TargetType> map1;
        std::map<int, TargetType> map2;
        auto converted_predicted = VectorConverter::convert_from_std<TargetType>(predicted, map1);
        auto converted_actual = VectorConverter::convert_from_std<TargetType>(actual, map2);
        return compute_classification_accuracy(converted_predicted, converted_actual);
    }

    //wrapper for std type vector representations
    //method for computing mean squared error from Eigen vectors of double, one with predicted values, the other with actual targets
    //mse = sum((prediction - actual_value)^2) / number_of_predictions
    template <Supported TargetType>
    double compute_mse(std::vector<TargetType> &predicted, std::vector<TargetType> &actual)
    {
        auto converted_predicted = VectorConverter::convert_from_std<TargetType>(predicted);
        auto converted_actual = VectorConverter::convert_from_std<TargetType>(actual);
        return compute_mse(converted_predicted, converted_actual);
    }
} // namespace metrics

#endif