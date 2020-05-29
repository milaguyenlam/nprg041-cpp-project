#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<double, double, double>> dataset;
    vector<chrono::time_point<chrono::steady_clock>> targets;
    for (double i = 0; i < 10; i++)
    {
        dataset.push_back(make_tuple(i, i, i));
        targets.push_back(chrono::steady_clock::now());
    }
}