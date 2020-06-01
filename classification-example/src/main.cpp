#include "../include/csv.h"
#include <iostream>
#include <cmath>
#include "../../include/model.hpp"

using namespace io;
using namespace std;

using classification_targets = vector<int>;
using classification_dataset = vector<tuple<int, float, int, int, double, int, int, int>>;

double accuracy(const classification_targets &predicted, const classification_targets &actual)
{
    int total_count = predicted.size();
    int correctly_classified = 0;
    for (int i = 0; i < total_count; i++)
    {
        if (predicted[i] == actual[i])
        {
            correctly_classified++;
        }
    }
    return (double)correctly_classified / (double)total_count;
}

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
    BinarySVM svm;

    ClassificationModelProxy<int, int, float, int, int, double, int, int, int>
        perceptron_proxy(perceptron);
    ClassificationModelProxy svm_proxy(svm, train_dataset, train_targets);

    perceptron_proxy.fit(train_dataset, train_targets);

    classification_targets perceptron_predictions = perceptron_proxy.predict(eval_dataset);
    classification_targets svm_predictions = perceptron_proxy.predict(eval_dataset);

    cout << "perceptron accuracy: " << accuracy(perceptron_predictions, eval_targets) << endl
         << "svm accuracy: " << accuracy(svm_predictions, eval_targets) << endl;
}