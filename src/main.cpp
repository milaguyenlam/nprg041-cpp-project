#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<double>> dataset;
    vector<double> targets;
    for (double i = 0; i < 30; i++)
    {
        dataset.push_back(make_tuple(i));
        targets.push_back(-i);
    }

    LinearRegression<double, double> model;
    model.fit(dataset, targets);
}