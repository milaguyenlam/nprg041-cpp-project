# MLLIB

MLLIB is a C++ library for basic machine learning, built on top of Eigen library.

## Description

- Implemented models are trained and used for predicting with data represented as Eigen::Matrix<> class.
- Conversion between certain std data types and Eigen classes is covered by a suite of conversion methods.
- You can also use proxy classes to fully automate the conversion process and use model "directly" with supported std types.

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

- supported conversions defined by template concepts
  `Supported` = {int, double, float, long, std::chrono::time_point<>}
  `SupportedClassificationTargets` = {int, long, string, char}
- available conversion methods (implemented in convert_utils.hpp)
