#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<double, double, double>> dataset;
    vector<double> targets;
    for (double i = 0; i < 10; i++)
    {
        dataset.push_back(make_tuple(i, i, i));
        targets.push_back(i);
    }
    auto edataset = convert_from_std(dataset);
    auto etargets = convert_from_std(targets);

    auto prediction_vector = convert_from_std(make_tuple(50, 50, 50));

    LinearRegression model;
    model.fit(edataset, etargets);
    cout << model.predict(prediction_vector) << endl;

    auto time = std::chrono::system_clock::now();
    auto double_time = convert_from(time);
    cout << double_time << endl;
    time = convert_to<chrono::system_clock>(double_time);
}