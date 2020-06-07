#include "../include/csv.h"
#include <iostream>
#include <cmath>
#include "../../include/model.hpp"

using namespace io;
using namespace std;
using namespace mllib;
using namespace metrics;

using classification_targets = vector<int>;
using classification_dataset = vector<tuple<int, float, int, int, double, int, int, int>>;

int main()
{
    CSVReader<9> train_parser("./resources/titanic-train.csv");
    CSVReader<9> eval_parser("./resources/titanic-eval.csv");
    int survived;
    int sex;
    float age;
    int family_count;
    int parch;
    double fair;
    int first_class_flag;
    int second_class_flag;
    int alone_flag;

    classification_targets train_targets;
    classification_dataset train_dataset;
    classification_targets eval_targets;
    classification_dataset eval_dataset;

    while (train_parser.read_row(survived, sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag))
    {
        train_targets.push_back(survived);
        train_dataset.push_back(make_tuple(sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag));
    }

    while (eval_parser.read_row(survived, sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag))
    {
        eval_targets.push_back(survived);
        eval_dataset.push_back(make_tuple(sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag));
    }
    Perceptron perceptron;
    perceptron.set_max_iterations(1000);
    BinarySVM svm;
    svm.set_max_iterations(5000);

    ClassificationModelProxy<int, int, float, int, int, double, int, int, int>
        perceptron_proxy(perceptron);
    ClassificationModelProxy svm_proxy(svm, train_dataset, train_targets);

    perceptron_proxy.fit(train_dataset, train_targets);

    classification_targets perceptron_predictions = perceptron_proxy.predict(eval_dataset);
    classification_targets svm_predictions = perceptron_proxy.predict(eval_dataset);

    cout << "perceptron accuracy: " << compute_classification_accuracy(perceptron_predictions, eval_targets) << endl
         << "svm accuracy: " << compute_classification_accuracy(svm_predictions, eval_targets) << endl;
}