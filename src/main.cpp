#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<int, double, double>> dataset;
    vector<int> targets;
    for (double i = 0; i < 100; i++)
    {
        dataset.push_back(make_tuple((int)i, i + 1, i + 2));
        targets.push_back((int)i);
    }

    LinearRegression<int, int, double, double> model;
    model.fit(dataset, targets);
}