#include "../include/csv.h"
#include <iostream>
#include <cmath>
#include "../../include/model.hpp"
#include "../../include/metrics.hpp"

using namespace io;
using namespace std;
using namespace mllib;
using namespace metrics;

using regression_targets = vector<double>;
using regression_dataset = vector<tuple<double, double, double, double, double, double>>;

int main()
{
    CSVReader<7> train_parser("./resources/train_data.csv");
    CSVReader<7> eval_parser("./resources/eval_data.csv");

    double f1;
    double f2;
    double f3;
    double f4;
    double f5;
    double f6;
    double target;

    regression_targets train_targets;
    regression_dataset train_dataset;
    regression_targets eval_targets;
    regression_dataset eval_dataset;

    while (train_parser.read_row(f1, f2, f3, f4, f5, f6, target))
    {
        train_targets.push_back(target);
        train_dataset.push_back(make_tuple(f1, f2, f3, f4, f5, f6));
    }

    while (eval_parser.read_row(f1, f2, f3, f4, f5, f6, target))
    {
        eval_targets.push_back(target);
        eval_dataset.push_back(make_tuple(f1, f2, f3, f4, f5, f6));
    }

    LinearRegression lin_reg;
    lin_reg.set_iterations(1000);

    RegressionModelProxy lin_reg_proxy(lin_reg, train_dataset, train_targets);

    regression_targets predictions = lin_reg_proxy.predict(eval_dataset);

    cout << "linear regression mean squared error: " << compute_mse(predictions, eval_targets) << endl;
}