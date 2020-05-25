#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<double, double, double>> dataset;
    vector<double> targets;
    for (double i = 0; i < 100; i++)
    {
        dataset.push_back(make_tuple(i, i + 1, i + 2));
        targets.push_back(i);
    }

    LinearRegression<double, double, double, double> model;
    model.fit(dataset, targets);
}