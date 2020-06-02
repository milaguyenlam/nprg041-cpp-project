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

//TODO: add include guards

namespace mllib
{
    using Vector1d = Eigen::Matrix<double, 1, 1>; //Eigen single value "vector" of double
    using Vector1i = Eigen::Matrix<int, 1, 1>;    //Eigen single value "vector" of int
    using namespace convert;

    //abstract class Model that all machine learning model implementations inherit from
    class Model
    {
    protected:
        //flag is model is trained or not
        bool trained = false;
        //used for checking if predict data correspond
        long int training_data_dimension;
        //training coefficients/weights
        Eigen::MatrixXd weights;
        //flag if padding is required
        bool apply_padding = true;
        //function padding given matrix with ones - creates new instance
        Eigen::MatrixXd pad_matrix_with_ones(const Eigen::MatrixXd &dataset) const
        {
            std::size_t rows = dataset.rows();
            std::size_t cols = dataset.cols();
            Eigen::MatrixXd padded_matrix(rows + 1, cols);
            for (size_t i = 0; i < rows + 1; i++)
            {
                for (size_t j = 0; j < cols; j++)
                {
                    //filling added row with ones
                    if (i == rows)
                    {
                        padded_matrix(i, j) = 1.0;
                    }
                    //copying values from given matrix
                    else
                    {
                        padded_matrix(i, j) = dataset(i, j);
                    }
                }
            }
            return padded_matrix;
        }
        //function padding given vector with one - creates new instance
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

    //abstract class representing classification model
    //uses Eigen data types (Eigen::Matrix<>)
    class ClassificationModel : public Model
    {
    protected:
        //actual algorithm implementations that child classes have to implement
        virtual void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &targets) = 0;
        virtual Vector1i make_prediction(const Eigen::VectorXd &vector) const = 0;
        virtual Eigen::VectorXi make_prediction(const Eigen::MatrixXd &vectors) const = 0;
        //flag if model only supports binary classification
        bool binary_classification = false;

    public:
        ~ClassificationModel() override
        {
        }
        //function that trains the model from data and targets
        //training data are always represented as Eigen matrix of double, a sample is represented as a matrix column
        //targets are represented as Eigen vector of ints, has to be in following format {0,...,N} where N is number of all classes
        //train data's number of columns has to equal the length of train targets
        void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXi &training_targets)
        {
            //checking if targets are in the right format
            check_targets(training_targets);
            //throws exception when training_data doesn't correspong to training_targets
            if (training_data.cols() != training_targets.size())
            {
                throw InvalidTrainingDataFormat();
            }
            training_data_dimension = training_data.rows();
            //applies padding to data if required, and passes it to virtual implementation of training_algorithm
            if (apply_padding)
            {
                auto padded_data = pad_matrix_with_ones(training_data);
                training_algorithm(padded_data, training_targets);
            }
            //passes raw data (not padded) to virtual implementation of training_algorithm
            else
            {
                training_algorithm(training_data, training_targets);
            }
            trained = true;
        }
        //function predicting a class based on input sample and pre-trained weights
        //a wrapper for actual implentation
        //sample is represented as an Eigen vector of double
        //sample's dimension has to correspond to that of training data's
        //this method can't be called without being trained before
        //returns single value "vector" of int
        Vector1i predict(const Eigen::VectorXd &vector) const
        {
            //throws exception if sample's dimension doesn't correspond
            if (training_data_dimension != vector.size())
            {
                throw InvalidPredictDataFormat();
            }
            //throws exception when called before being trained
            if (!trained)
            {
                throw PredictBeforeFit();
            }
            //applying padding if required and passing it to virtual implementation
            if (apply_padding)
            {
                auto padded_vector = pad_vector_with_ones(vector);
                return make_prediction(padded_vector);
            }
            //passing raw sample (unpadded)
            else
            {
                return make_prediction(vector);
            }
        }
        //function predicting a class based on input sample and pre-trained weights
        //a wrapper for actual implentation
        //samples are represented as an Eigen matrix of double (single sample as a matrix column)
        //sample's dimension has to correspond to that of training data's
        //this method can't be called without being trained before
        //returns Eigen vector of int, where i-th row corresponds to i-th sample prediction
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
        //check if targets are in the right format, that is {0,..,N} where N is number of classes
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
                //throws exception when targets are in incorrect format
                if (!found_target_labels.contains(i))
                {
                    throw InvalidTargetFormat();
                }
            }
            //throws an exception when there aren't 2 classes and model only supports binary classification
            if ((elements_count != 2) && binary_classification)
            {
                throw BinaryLabelsExpected();
            }
        }
    };

    //abstract class representing regression model
    //uses Eigen data types (Eigen::Matrix<>)
    class RegressionModel : public Model
    {
    protected:
        //actual algorithm implementations that child classes have to implement
        virtual void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXd &targets) = 0;
        virtual Vector1d make_prediction(const Eigen::VectorXd &vector) const = 0;
        virtual Eigen::VectorXd make_prediction(const Eigen::MatrixXd &vectors) const = 0;

    public:
        virtual ~RegressionModel()
        {
        }
        //function that trains the model from data and targets
        //training data are always represented as Eigen matrix of double, a sample is represented as a matrix column
        //targets are represented as Eigen vector of double
        //train data's number of columns has to equal the length of train targets
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
        //function predicting a class based on input sample and pre-trained weights
        //a wrapper for actual implentation
        //sample is represented as an Eigen vector of double
        //sample's dimension has to correspond to that of training data's
        //this method can't be called without being trained before
        //returns single value "vector" of double
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
        //function predicting a class based on input sample and pre-trained weights
        //a wrapper for actual implentation
        //samples are represented as an Eigen matrix of double (single sample as a matrix column)
        //sample's dimension has to correspond to that of training data's
        //this method can't be called without being trained before
        //returns Eigen vector of int, where i-th row corresponds to i-th sample prediction
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
    };

    //linear regression model implementation
    //uses SGD to train weights
    //default values:
    //regularization lambda = 0
    //iterations = 100
    //sgd learning rate = 0.01
    class RidgeRegression : public RegressionModel
    {
    private:
        //L2 regularization
        double lambda = 0;
        //number of iterations training algorithm should take
        std::size_t iterations = 100;
        //sgd learning rate
        double learning_rate = 0.01;

    protected:
        //Ridge regression training algorithm actual implementation
        void
        training_algorithm(const Eigen::MatrixXd &data, const Eigen::VectorXd &targets) override
        {
            std::size_t data_dimension = data.rows();
            std::size_t data_length = data.cols();
            weights = Eigen::MatrixXd(data_dimension, 1).setZero(); //initializing weights = 0
            for (std::size_t iter = 0; iter < iterations; iter++)
            {
                //iterate through every input sample
                for (std::size_t i = 0; i < data_length; i++)
                {
                    Eigen::VectorXd input_vector = data.col(i);                          //get i-th sample
                    double target = targets(i);                                          //get its corresponding target
                    auto prediction = (make_prediction(input_vector))(0, 0);             //compute prediction
                    auto regularization = lambda * weights;                              //compute l2 regularization
                    auto error = prediction - target;                                    //compute error
                    auto step = learning_rate * (error * input_vector) + regularization; //compute sgd step
                    weights = weights - step;                                            //update weights/coeffs
                }
            }
        }
        //computing prediction from an input sample and pretrained weights
        Vector1d make_prediction(const Eigen::VectorXd &vector) const override
        {
            return vector.transpose() * weights;
        }
        //computing predictions from input samples and pretrained weights
        Eigen::VectorXd make_prediction(const Eigen::MatrixXd &vectors) const override
        {
            return vectors.transpose() * weights;
        }

    public:
        ~RidgeRegression() override
        {
        }
        //default: 0
        //sets lambda
        RidgeRegression &with_L2_regulatization(double new_lambda)
        {
            lambda = new_lambda;
            return *this;
        }
        //default: 100
        //sets number of iterations training should do
        RidgeRegression &set_iterations(std::size_t new_iterations)
        {
            iterations = new_iterations;
            return *this;
        }
        //default: 0.01
        //sets learning rate
        RidgeRegression &set_learning_rate(double alpha)
        {
            learning_rate = alpha;
            return *this;
        }
    };

    //Perceptron model implementation
    //uses SGD to train weights
    //default values:
    //sgd learning rate = 0.01
    //max iterations = 100
    class Perceptron : public ClassificationModel
    {
    private:
        double learning_rate = 0.01;
        //number of maximum iterations training algorithm should do before returning without being separated
        std::size_t iterations = 100;
        //flag if dataset was succesfully separated
        bool dataset_separated = false;

    protected:
        //actual perceptron training algorithm implementation
        void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &targets) override
        {
            std::size_t data_lenght = dataset.cols();
            std::size_t data_dimension = dataset.rows();
            weights = Eigen::MatrixXd(data_dimension, 1).setZero(); //initializing weights

            for (size_t it = 0; it < iterations; it++)
            {
                bool incorrect_classification = false; //set flag that indicates an incorrect classification
                //iterate through every input sample
                for (size_t i = 0; i < data_lenght; i++)
                {
                    Eigen::VectorXd vector = dataset.col(i);         //get i-th sample
                    auto target = targets[i];                        //get its corresponding target
                    auto prediction = make_prediction(vector)(0, 0); //compute prediction
                    //check if prediction was made correctly, if not update the weights
                    if (prediction != target)
                    {
                        incorrect_classification = true;
                        auto error = target - prediction;                     //compute error
                        weights = weights + (learning_rate * error) * vector; //update weights with sgd step
                    }
                }
                //if every sample was classified correctly end the algorithm and set separation flag to true
                if (!incorrect_classification)
                {
                    dataset_separated = true;
                    return;
                }
            }
        };
        //computing prediction from an input sample and pretrained weights
        //outputs Eigen single value vector of int from {0,1}
        Vector1i make_prediction(const Eigen::VectorXd &vector) const override
        {
            auto dot = (vector.transpose() * weights)(0, 0);
            Vector1i result;
            //return 1 if dot product is above the separation plane, 0 otherwise
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
        //computing predictions from input samples and pretrained weights
        //outputs Eigen vector of int from {0,1}
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
        //if dataset was correctly separated
        bool was_dataset_separated()
        {
            return dataset_separated;
        }
        //default: 100
        //setting max iterations that training algorithm should do before ending without being separated
        Perceptron &set_max_iterations(std::size_t new_iterations)
        {
            iterations = new_iterations;
            return *this;
        }
        //default: 0.01
        //setting sgd learning rate
        Perceptron &set_learning_rate(double rate)
        {
            learning_rate = rate;
            return *this;
        }
    };

    //binary SVM model implementation
    //uses SMO to train the weights
    //default values:
    //regularization parameter = 1
    //numerical tolerance = 0.0001
    //maximum passes = 5
    //maximum iteration = train_sample_size * train_sample_size
    class BinarySVM : public ClassificationModel
    {
    protected:
        void training_algorithm(const Eigen::MatrixXd &dataset, const Eigen::VectorXi &input_targets) override
        {
            Eigen::VectorXi targets = change_target_values(input_targets); //preprocess targets
            train_targets = targets;                                       //save targets to internal train_targets variable
            train_data = dataset.replicate(1, 1);                          //save dataset to internal train_data variable
            std::size_t dataset_size = dataset.cols();
            weights = Eigen::VectorXd(dataset_size).setZero();                                     //initializing weights (representing Lagrange multipliers)
            b = 0;                                                                                 //initializing bias
            compute_kernel_values(dataset);                                                        //computing kernel values from given dataset
            std::size_t passes = 0;                                                                //(how many passes in a row)
            max_iterations = (max_iterations != 0) ? dataset_size * dataset_size : max_iterations; //initializing maximum number of iterations (if not set by a user then dataset_size * dataset_size)
            std::size_t iterations = 0;                                                            //(how many iterations done)
            while (passes < max_passes)
            {
                //end algorithm execution when number of iterations done exceeds max_iterations
                if (iterations > max_iterations)
                {
                    return;
                }
                std::size_t a_changed = 0; //initializes number of weights/lagrange multipliers changed
                std::size_t a_length = weights.rows();
                //iterate through every sample in dataset
                for (std::size_t i = 0; i < a_length; i++)
                {
                    double error = predict_value(dataset.col(i)) - (double)targets[i]; //compute error for i-th sample
                    //checking KKT conditions
                    if (((weights(i, 0) < regularization_param) && (targets[i] * error < -numerical_tolerance)) || ((weights(i, 0) > 0) && (targets[i] * error > numerical_tolerance)))
                    {
                        //selecting index j != i
                        std::size_t j = std::rand() % (a_length - 1);
                        while (j == i)
                        {
                            j = std::rand() % (a_length - 1);
                        }
                        //computing second derivative of the loss
                        double second_der = 2 * kernel_values(i, j) - kernel_values(i, i) - kernel_values(j, j);

                        if (second_der >= -numerical_tolerance)
                        {
                            continue;
                        }
                        //compute error for j-th sample
                        double error_j = predict_value(dataset.col(j)) - targets[j];
                        //computing new j-th weight value
                        double weight_j = weights(j, 0) - (targets[j] * (error - error_j)) / second_der;

                        double L;
                        double H;
                        //computing LO, HI interval
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
                        //clamping weight_j to [L,H] inverval
                        std::clamp(weight_j, L, H);

                        //if clip is too narrow, continue with next i
                        if ((H - L < numerical_tolerance) || (std::abs(weight_j - weights(j, 0)) < numerical_tolerance))
                        {
                            continue;
                        }

                        //computing new i-th weight value
                        double weight_i = weights(i, 0) - (targets[i] * targets[j] * (weight_j - weights(j, 0)));
                        //computing possible new bias values
                        double b_i = b - error - (targets[i] * (weight_i - weights(i, 0)) * kernel_values(i, i)) - (targets[j] * (weight_j - weights(j, 0)) * kernel_values(j, i));
                        double b_j = b - error_j - (targets[i] * (weight_i - weights(i, 0)) * kernel_values(i, j)) - (targets[j] * (weight_j - weights(j, 0)) * kernel_values(j, j));

                        //choosing which value to update bias with
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
                        //update j-th weight and i-th weight with pre-computed values
                        weights(i, 0) = weight_i;
                        weights(j, 0) = weight_j;
                        //weights has been updated
                        a_changed++;
                    }
                }
                iterations++;
                //if weights changed then null the passes, increment otherwise
                passes = (a_changed > 0) ? 0 : passes + 1;
            }
            //sets converge flag if algorithm finishes by not passing (passes < max_passes) condition
            alg_converged = true;
        }
        //computing prediction from an input sample and pretrained weights
        //outputs Eigen single value vector of int from {0,1}
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
        //computing predictions from input samples and pretrained weights
        //outputs Eigen vector of int from {0,1}
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
        //smo regularization parameter
        double regularization_param = 1;
        //smo numerical tolerance
        double numerical_tolerance = 0.0001;
        //bias
        double b = 0;
        //flag if algorithm converged to a optimal solution
        bool alg_converged = false;
        //maximum number of iterations training algorithm should do before returning without converging
        std::size_t max_iterations = 0;
        //maximum number of sequential passes (iteration where weights weren't updated) before returning
        std::size_t max_passes = 5;
        //pre computed kernel values
        Eigen::MatrixXd kernel_values;
        //saved train targets
        Eigen::VectorXi train_targets;
        //saved train data
        Eigen::MatrixXd train_data;

        //helper method that computes linear kernel values from given dataset represented as Eigen matrix of double
        //saves it into private variable kernel_values
        void compute_kernel_values(const Eigen::MatrixXd &dataset)
        {
            std::size_t data_size = dataset.cols();
            kernel_values = Eigen::MatrixXd(data_size, data_size);
            //compute dot product of every pair of input samples
            //saves into matrix : matrix[i,j] = i-th sample * j-th sample
            for (size_t i = 0; i < data_size; i++)
            {
                for (size_t j = 0; j < data_size; j++)
                {
                    Eigen::VectorXd vector_i = dataset.col(i);                //get i-th sample
                    Eigen::VectorXd vector_j = dataset.col(j);                //get j-th sample
                    kernel_values(i, j) = compute_kernel(vector_i, vector_j); //computes kernel value and saves it into kernel_values variable
                }
            }
        }

        //computation of linear kernel value (dot product) for pair of vectors
        //takes 2 Eigen vectors of double as parameters and outputs double
        double compute_kernel(const Eigen::VectorXd &vector, const Eigen::VectorXd &vector2) const
        {
            return (vector.transpose() * vector2)(0, 0); //computes dot product and "casts" it to double
        }

        //computing prediction value
        //outputs double
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
        //training target vector preprocessing method that changes 0 -> -1 - creates new instance
        //training algorithm expects labels in following format - {-1, 1}
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
        //default: 1
        //setting regularization parameter
        BinarySVM &set_regularization_param(double reg_param)
        {
            regularization_param = reg_param;
            return *this;
        }
        //default: 0.0001
        //setting numerical tolerance
        BinarySVM &set_tolerance(double tolerance)
        {
            numerical_tolerance = tolerance;
            return *this;
        }
        //default: 5
        //setting maximum number of sequential passes (iteration where weights weren't updated) before returning
        BinarySVM &set_max_passes(std::size_t passes)
        {
            max_passes = passes;
            return *this;
        }
        //default: dataset_size * dataset_size, where dataset size is number of samples in given dataset
        //maximum number of iterations training algorithm should do before returning without converging
        BinarySVM &set_max_iterations(std::size_t iterations)
        {
            max_iterations = iterations;
            return *this;
        }
        //if smo algorithm converged
        bool converged()
        {
            return alg_converged;
        }
    };

    //Proxy class that automates conversion process from std types to Eigen data types and back
    //wraps a regression model instance
    //single sample is represented by std::tuple<TupleTypes...>, whole dataset by std::vector of those
    //takes TargetType and variadic TupleTypes as template parameter
    template <Supported TargetType, Supported... TupleTypes>
    class RegressionModelProxy
    {
    private:
        //reference to a actual model instance
        RegressionModel &model;

    public:
        //takes model instance reference
        RegressionModelProxy(RegressionModel &model_to_wrap) : model(model_to_wrap)
        {
        }
        //constructor overload that trains the model right after construction
        //takes model instance reference
        //also helpful for type deduction
        RegressionModelProxy(RegressionModel &model_to_wrap, const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets) : model(model_to_wrap)
        {
            fit(data, targets);
        }
        //takes dataset and targets represented in std types, converts them into Eigen data types and trains the model
        void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets)
        {
            auto converted_data = convert_from_std<TupleTypes...>(data);
            auto converted_targets = VectorConverter::convert_from_std<TargetType>(targets);
            model.fit(converted_data, converted_targets);
        }
        //takes sample represented as std::tuple<TupleTypes...>, converts it into Eigen vector of double and passes it to the model
        //converts model output to specified TargetType and outputs that value
        TargetType predict(const std::tuple<TupleTypes...> &tuple)
        {
            auto converted_vector = convert_from_std<TupleTypes...>(tuple);
            Vector1d value = model.predict(converted_vector);
            TargetType ret_value = SingleValueConverter::convert_to<TargetType>(value[0]);
            return ret_value;
        }
        //takes sample set represented as std::vector<std::tuple<TupleTypes...>>, converts it into Eigen matrix of double and passes it to the model
        //converts model output to specified std::vector<TargetType> and outputs that
        std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples)
        {
            auto converted_matrix = convert_from_std<TupleTypes...>(tuples);
            auto value_vector = model.predict(converted_matrix);
            std::vector<TargetType> ret_values = VectorConverter::convert_to_std<TargetType>(value_vector);
            return ret_values;
        }
    };

    //Proxy class that automates conversion process from std types to Eigen data types and back
    //wraps a regression model instance
    //single sample is represented by std::tuple<TupleTypes...>, whole dataset by std::vector of those
    //takes TargetType and variadic TupleTypes as template parameter
    template <SupportedClassificationTargets TargetType, Supported... TupleTypes>
    class ClassificationModelProxy
    {
    private:
        ClassificationModel &model;          //reference to a actual model instance
        std::map<int, TargetType> label_map; //map converting prediction outputs to specified TargetType

    public:
        //takes model instance reference
        ClassificationModelProxy(ClassificationModel &model_to_wrap) : model(model_to_wrap)
        {
        }
        //constructor overload that trains the model right after construction
        //takes model instance reference
        //also helpful for type deduction
        ClassificationModelProxy(ClassificationModel &model_to_wrap, const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets) : model(model_to_wrap)
        {
            fit(data, targets);
        }
        //takes dataset and targets represented in std types, converts them into Eigen data types and trains the model
        void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets)
        {
            auto converted_data = convert_from_std(data);
            auto converted_targets = VectorConverter::convert_from_std(targets, label_map);
            model.fit(converted_data, converted_targets);
        }
        //takes sample represented as std::tuple<TupleTypes...>, converts it into Eigen vector of double and passes it to the model
        //converts model output to specified TargetType and outputs that value
        TargetType predict(const std::tuple<TupleTypes...> &tuple)
        {
            auto converted_vector = convert_from_std<TupleTypes...>(tuple);
            auto value = model.predict(converted_vector);
            TargetType ret_value = label_map.find(value[0])->second;
            return ret_value;
        }
        //takes sample set represented as std::vector<std::tuple<TupleTypes...>>, converts it into Eigen matrix of double and passes it to the model
        //converts model output to specified std::vector<TargetType> and outputs that
        std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples)
        {
            auto converted_matrix = convert_from_std<TupleTypes...>(tuples);
            auto values = model.predict(converted_matrix);
            auto ret_values = VectorConverter::convert_to_std(values, label_map);
            return ret_values;
        }
    };

    double compute_classification_accuracy(Eigen::VectorXi &predicted, Eigen::VectorXi &actual)
    {
        std::size_t predicted_size = predicted.size();
        std::size_t actual_size = actual.size();
        if (predicted_size != actual_size)
        {
            //throw exception
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

    double compute_mse(Eigen::VectorXd &predicted, Eigen::VectorXd &actual)
    {
        return 0.75;
    }

    template <SupportedClassificationTargets TargetType>
    double compute_classification_accuracy(std::vector<TargetType> &predicted, std::vector<TargetType> &actual)
    {
        std::map<int, TargetType> map1;
        std::map<int, TargetType> map2;
        auto converted_predicted = VectorConverter::convert_from_std<TargetType>(predicted, map1);
        auto converted_actual = VectorConverter::convert_from_std<TargetType>(actual, map2);
        return compute_classification_accuracy(converted_predicted, converted_actual);
    }

    template <Supported TargetType>
    double compute_mse(std::vector<TargetType> &predicted, std::vector<TargetType> &actual)
    {
        auto converted_predicted = VectorConverter::convert_from_std<TargetType>(predicted);
        auto converted_actual = VectorConverter::convert_from_std<TargetType>(predicted);
        return compute_mse(converted_predicted, converted_actual);
    }
} // namespace mllib
