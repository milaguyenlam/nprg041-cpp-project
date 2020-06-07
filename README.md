# MLLIB

MLLIB is a C++ library for basic machine learning, built on top of Eigen library.

## Description

- Implemented models are trained and used for predicting with data represented as Eigen::Matrix<> class.
- Conversion between certain std data types and Eigen classes is covered by a suite of conversion methods.
- You can also use proxy classes to fully automate the conversion process and use model "directly" with supported std types.

## Installation

Mllib is a template library defined in the headers. You can use header files (in ./include folder of the project) right away (just copy them to your include path).
Note that, there has to be Eigen/Core library present in include path as well. (As it is in the project).

## Usage

- example demonstrating basic usage of the library

```cpp
    vector<tuple<double, double>> dataset;
    vector<string> targets;
    create_data(dataset, targets);

    vector<tuple<double, double>> to_predict;
    vector<string> test_targets;
    create_testing_data(to_predict, test_targets);

    //STEP 1: model initialization and modification
    Perceptron perc;
    //modification is done by model's methods (implemented with fluent interface)
    perc
        .set_learning_rate(0.001)
        .set_max_iterations(1200);
    //wrapping the model with proxy class to automate data conversion process
    ClassificationModelProxy<string, double, double> perc_proxy(perc);

    //STEP 2: fitting the model
    perc_proxy.fit(dataset, targets);

    //STEP 3: predicting target values
    //by calling predict() method via proxy instance, output predictions will be converted to target type given in fit() method
    auto perc_predicted_values = perc_proxy.predict(to_predict); //returns vector<string> type with predicted values

    BinarySVM svm;
    //also calls fit() method when passing training data in proxy class constructor
    ClassificationModelProxy svm_proxy(svm, dataset, targets);
    auto svm_predicted_values = svm_proxy.predict(to_predict);

    //STEP 4: computing metrics
    //you can use built-in method for computing accuracy and mse
    cout << "svm model accuracy: " << compute_classification_accuracy(svm_predicted_values, test_targets) * 100.0 << "%" << endl
         << "perceptron model accuracy: " << compute_classification_accuracy(perc_predicted_values, test_targets) * 100.0 << "%" << endl;
```

- you can also convert data using convert methods implemented in convert_utils.hpp

```cpp
    vector<tuple<double, double>> dataset;
    vector<string> targets;
    create_data(dataset, targets);

    vector<tuple<double, double>> to_predict;
    vector<string> test_targets;
    create_testing_data(to_predict, test_targets);

    //note that full method call should look like this: convert::convert_from_std<double, double>(dataset);
    //but you don't have to you that version if your compiler supports type deduction
    Eigen::MatrixXd dataset_matrix = convert_from_std(dataset);
    Eigen::MatrixXd to_predict_matrix = convert_from_std(to_predict);

    //when converting categorical targets, you have to pass an empty map<int, TargetType> to the following convert method()
    //passed map will get filled with conversion that will be useful for converting prediction values back to their initial type
    std::map<int, string> converter;
    Eigen::VectorXi target_vector = VectorConverter::convert_from_std(targets, converter);

    Perceptron perceptron;
    perceptron.fit(dataset_matrix, target_vector);

    Eigen::VectorXi prediction_values = perceptron.predict(to_predict_matrix);

    //converting output Eigen vector to initial std representation
    vector<string> std_predictions = VectorConverter::convert_to_std(prediction_values, converter);
```

## Documentation

- supported conversions defined by template concepts:

  - `Supported` = {int, double, float, long, std::chrono::time_point<>
  - `SupportedClassificationTargets` = {int, long, string, char}

- available conversion methods (implemented in convert_utils.hpp), divided into static classes - `namespace convert`

  ```cpp
    //static class used for converting between double and Supported types (both directions)
    class SingleValueConverter
    {
    public:
        template <Supported From>
        static double convert_from(From value);

        //converting from time_point class
        template <class Clock, class Duration>
        static double convert_from(const std::chrono::time_point<Clock, Duration> &value);

        template <Supported To>
        static To convert_to(double value);
    };

    //class used for converting std::vector of Supported (or SupportedClassificationTargets) from/to Eigen data types
    //uses From, To of Supported types as type paramaters
    class VectorConverter
    {
    public:
        //overload for a single vector
        template <Supported From>
        static Eigen::VectorXd convert_from_std(const std::vector<From> &vector);

        //overload for a matrix represented as nested std::vector<std::vector>
        template <Supported From>
        static Eigen::MatrixXd convert_from_std(const std::vector<std::vector<From>> &matrix);

        //converting Eigen vector type to std::vector of "To" type
        template <Supported To>
        static std::vector<To> convert_to_std(const Eigen::VectorXd &vector);

        //function used for converting std::vector to Eigen vector type (integer)
        //maps contents as classification labels to {0,1,...,N}, where N is number of values inside given vector
        //also fills a std::map with "back direction" conversions, given map has to be empty
        template <SupportedClassificationTargets TargetType>
        static Eigen::VectorXi convert_from_std(const std::vector<TargetType> &vector,std::map<int, TargetType> &converter);

        //used for converting classification model output type (Eigen int vector) to std::vector<TargetType> using map filled with int->TargetType conversions
        template <SupportedClassificationTargets TargetType>
        static std::vector<TargetType> convert_to_std(const Eigen::VectorXi &vector, const std::map<int, TargetType> &converter);
    };

    //methods for std::tuple of Supported types conversion
    //putting them into a class did not work, see https://gcc.gnu.org/bugzilla//show_bug.cgi?id=79917
    //they are left as global methods as a temporary solution

    //converting from std::tuple to Eigen vector class
    template <Supported... From>
    Eigen::VectorXd convert_from_std(const std::tuple<From...> &tuple);

    //overload for matrix represented as std::vector<tuple<>>
    template <Supported... From>
    Eigen::MatrixXd convert_from_std(const std::vector<std::tuple<From...>> &tuples);
  ```

- available machine learning models - `namespace mllib`

  - `RegressionModel`

    - abstract class that regression models inherit from
    - uses Eigen data types
    - defines basic API

      ```cpp
      //function that trains the model from data and targets
      //training data are always represented as Eigen matrix of double, a sample is represented as a matrix column
      //targets are represented as Eigen vector of double
      //train data's number of columns has to equal the length of train targets
      void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXd &training_targets);

      //function predicting a class based on input sample and pre-trained weights
      //a wrapper for actual implentation
      //sample is represented as an Eigen vector of double
      //sample's dimension has to correspond to that of training data's
      //this method can't be called without being trained before
      //returns single value "vector" of double
      Vector1d predict(const Eigen::VectorXd &vector) const;

      //function predicting a class based on input sample and pre-trained weights
      //a wrapper for actual implentation
      //samples are represented as an Eigen matrix of double (single sample as a matrix column)
      //sample's dimension has to correspond to that of training data's
      //this method can't be called without being trained before
      //returns Eigen vector of int, where i-th row corresponds to i-th sample prediction
      Eigen::VectorXd predict(const Eigen::MatrixXd &vectors) const;
      ```

  - `ClassificationModel`

    - abstract class that classification models inherit from
    - uses Eigen data types
    - defines basic API

      ```cpp
      //function that trains the model from data and targets
      //training data are always represented as Eigen matrix of double, a sample is represented as a matrix column
      //targets are represented as Eigen vector of ints, has to be in following format {0,...,N} where N is number of all classes
      //train data's number of columns has to equal the length of train targets
      void fit(const Eigen::MatrixXd &training_data, const Eigen::VectorXi &training_targets);

      //function predicting a class based on input sample and pre-trained weights
      //a wrapper for actual implentation
      //sample is represented as an Eigen vector of double
      //sample's dimension has to correspond to that of training data's
      //this method can't be called without being trained before
      //returns single value "vector" of int
      Vector1i predict(const Eigen::VectorXd &vector) const;

      //function predicting a class based on input sample and pre-trained weights
      //a wrapper for actual implentation
      //samples are represented as an Eigen matrix of double (single sample as a matrix column)
      //sample's dimension has to correspond to that of training data's
      //this method can't be called without being trained before
      //returns Eigen vector of int, where i-th row corresponds to i-th sample prediction
      Eigen::VectorXi predict(const Eigen::MatrixXd &vectors) const;
      ```

  - `BinarySVM`

    - model using simplified SMO algorithm to train weights/Lagrange multipliers
    - algorithm taken from: http://cs229.stanford.edu/materials/smo.pdf
    - training algorithm doesn't always have to converge
    - inherits from `ClassificationModel` class, therefore inherits its fit() and predict() methods
    - can be modified by following methods (fluent interface/ method chaining)

      ```cpp
      //default: 1
      //setting regularization parameter
      BinarySVM &set_regularization_param(double reg_param);

      //default: 0.0001
      //setting numerical tolerance
      BinarySVM &set_tolerance(double tolerance);

      //default: 5
      //setting maximum number of sequential passes (iteration where weights weren't updated) before returning
      BinarySVM &set_max_passes(std::size_t passes);

      //default: dataset_size * dataset_size, where dataset size is number of samples in given dataset
      //maximum number of iterations training algorithm should do before returning without converging
      BinarySVM &set_max_iterations(std::size_t iterations);
      ```

    - information about training converging while training can be accessed by `bool converged()` method

  - `Perceptron`

    - model using SGD perceptron algorithm to train weights
    - training algorithm doesn't always have to converge (when dataset is not linearly separable)
    - inherits from `ClassificationModel` class, therefore inherits its fit() and predict() methods
    - can be modified by following methods (fluent interface/ method chaining)

      ```cpp
      //default: 100
      //setting max iterations that training algorithm should do before ending without being separated
      Perceptron &set_max_iterations(std::size_t new_iterations);

      //default: 0.01
      //setting sgd learning rate
      Perceptron &set_learning_rate(double rate);
      ```

    - information about convergence during training can be accessed by `bool was_dataset_separated()` method

  - `LinearRegression`

    - model using SGD(regular) linear regression algorithm (additionally with L2 regularization) to train weights
    - training algorithm doesn't always have to converge (when dataset is not linearly separable)
    - inherits from `RegressionModel` class, therefore inherits its fit() and predict() methods
    - can be modified by following methods (fluent interface/ method chaining)

      ```cpp
      //default: 0
      //sets lambda
      LinearRegression &with_L2_regulatization(double new_lambda);

      //default: 100
      //sets number of iterations training should do
      LinearRegression &set_iterations(std::size_t new_iterations);

      //default: 0.01
      //sets learning rate
      LinearRegression &set_learning_rate(double alpha);
      ```

- proxy classes for conversion automation - `namespace mllib`

  - `RegressionModelProxy`

    ```cpp
    //Proxy class that automates conversion process from std types to Eigen data types and back
    //wraps a regression model instance
    //single sample is represented by std::tuple<TupleTypes...>, whole dataset by std::vector of those
    //takes TargetType and variadic TupleTypes as template parameter
    template <Supported TargetType, Supported... TupleTypes>
    class RegressionModelProxy
    {
    public:
        //takes model instance reference
        RegressionModelProxy(RegressionModel &model_to_wrap) : model(model_to_wrap);

        //constructor overload that trains the model right after construction
        //takes model instance reference
        //also helpful for type deduction
        RegressionModelProxy(RegressionModel &model_to_wrap, const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets) : model(model_to_wrap);

        //takes dataset and targets represented in std types, converts them into Eigen data types and trains the model
        void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets);

        //takes sample represented as std::tuple<TupleTypes...>, converts it into Eigen vector of double and passes it to the model
        //converts model output to specified TargetType and outputs that value
        TargetType predict(const std::tuple<TupleTypes...> &tuple);

        //takes sample set represented as std::vector<std::tuple<TupleTypes...>>, converts it into Eigen matrix of double and passes it to the model
        //converts model output to specified std::vector<TargetType> and outputs that
        std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples);
    };
    ```

  - `ClassificationModelProxy`

    ```cpp
    //Proxy class that automates conversion process from std types to Eigen data types and back
    //wraps a regression model instance
    //single sample is represented by std::tuple<TupleTypes...>, whole dataset by std::vector of those
    //takes TargetType and variadic TupleTypes as template parameter
    template <SupportedClassificationTargets TargetType, Supported... TupleTypes>
    class ClassificationModelProxy
    {
    public:
        //takes model instance reference
        ClassificationModelProxy(ClassificationModel &model_to_wrap) : model(model_to_wrap);

        //constructor overload that trains the model right after construction
        //takes model instance reference
        //also helpful for type deduction
        ClassificationModelProxy(ClassificationModel &model_to_wrap, const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets) : model(model_to_wrap);

        //takes dataset and targets represented in std types, converts them into Eigen data types and trains the model
        void fit(const std::vector<std::tuple<TupleTypes...>> &data, const std::vector<TargetType> &targets);

        //takes sample represented as std::tuple<TupleTypes...>, converts it into Eigen vector of double and passes it to the model
        //converts model output to specified TargetType and outputs that value
        TargetType predict(const std::tuple<TupleTypes...> &tuple);

        //takes sample set represented as std::vector<std::tuple<TupleTypes...>>, converts it into Eigen matrix of double and passes it to the model
        //converts model output to specified std::vector<TargetType> and outputs that
        std::vector<TargetType> predict(const std::vector<std::tuple<TupleTypes...>> &tuples);
    };

    ```

- basic metrics measuring methods - `namespace metrics`

  ```cpp
    //method computing accuracy from Eigen vectors of int, one with predicted values, the other with actual targets
    //accuracy = correctly_predicted / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    double compute_classification_accuracy(Eigen::VectorXi &predicted, Eigen::VectorXi &actual);

    //method for computing mean squared error from Eigen vectors of double, one with predicted values, the other with actual targets
    //mse = sum((prediction - actual_value)^2) / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    double compute_mse(Eigen::VectorXd &predicted, Eigen::VectorXd &actual);

    //wrapper for std type vector representations
    //method computing accuracy from Eigen vectors of int, one with predicted values, the other with actual targets
    //accuracy = correctly_predicted / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    template <SupportedClassificationTargets TargetType>
    double compute_classification_accuracy(std::vector<TargetType> &predicted, std::vector<TargetType> &actual);

    //wrapper for std type vector representations
    //method for computing mean squared error from Eigen vectors of double, one with predicted values, the other with actual targets
    //mse = sum((prediction - actual_value)^2) / number_of_predictions
    //throws exception data predicted vector's lenght doesn't equal actual's lenght
    template <Supported TargetType>
    double compute_mse(std::vector<TargetType> &predicted, std::vector<TargetType> &actual);
  ```
