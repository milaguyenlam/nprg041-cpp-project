//#include "../include/model.hpp"
#include <iostream>
#include <tuple>
#include "../include/convert_util.hpp"

using namespace std;

int main()
{
    vector<tuple<double, double, double>> dataset;
    vector<float> targets;
    for (double i = 0; i < 10; i++)
    {
        dataset.push_back(make_tuple(i, i, i));
        targets.push_back(i);
    }

    auto time = convert_to<chrono::time_point<chrono::high_resolution_clock, chrono::seconds>>(1554.0);
    auto edataset = convert_from_std(dataset);
    auto etargets = convert_from_std(targets);

    auto matrix = convert_to_std<float>(edataset);

    auto vector = make_tuple(50, 50, 50);
}