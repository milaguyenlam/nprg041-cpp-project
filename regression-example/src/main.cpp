#include "../include/csv.h"
#include <iostream>
#include <cmath>
#include "../../include/model.hpp"

using namespace io;
using namespace std;

using regression_targets = vector<float>;
using regression_dataset = vector<tuple<int, int, int, int, double, int, int, int>>;

double mean_error(const regression_targets &predicted, const regression_targets &actual)
{
    std::size_t total_count = predicted.size();
    float error_sum = 0;
    for (size_t i = 0; i < total_count; i++)
    {
        error_sum += std::abs(predicted[i] - actual[i]);
    }
    return error_sum / (double)total_count;
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

    regression_targets train_targets;
    regression_dataset train_dataset;
    regression_targets eval_targets;
    regression_dataset eval_dataset;

    while (train_parser.read_row(survived, sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag))
    {
        train_targets.push_back(age);
        train_dataset.push_back(make_tuple(survived, sex, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag));
    }

    while (eval_parser.read_row(survived, sex, age, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag))
    {
        eval_targets.push_back(age);
        eval_dataset.push_back(make_tuple(survived, sex, family_count, parch, fair, first_class_flag, second_class_flag, alone_flag));
    }

    RidgeRegression lin_reg;

    RegressionModelProxy lin_reg_proxy(lin_reg, train_dataset, train_targets);

    regression_targets predictions = lin_reg_proxy.predict(eval_dataset);

    cout << "linear regression mean error: " << mean_error(predictions, eval_targets) << endl;
}