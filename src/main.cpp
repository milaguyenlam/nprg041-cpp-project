#include "../include/model.hpp"
#include <iostream>
#include <tuple>

using namespace std;

int main()
{
    vector<tuple<double, double>> dataset;
    vector<string> targets;
    dataset.push_back(make_tuple(0.0, 0.0));
    dataset.push_back(make_tuple(0.0, 3.0));
    dataset.push_back(make_tuple(4.0, 0.0));
    dataset.push_back(make_tuple(0.0, 3.0));

    dataset.push_back(make_tuple(7.0, 8.0));
    dataset.push_back(make_tuple(6.0, 9.0));
    dataset.push_back(make_tuple(8.0, 8.0));
    dataset.push_back(make_tuple(10.0, 10.0));

    targets.push_back("A");
    targets.push_back("A");
    targets.push_back("A");
    targets.push_back("A");

    targets.push_back("B");
    targets.push_back("B");
    targets.push_back("B");
    targets.push_back("B");

    vector<tuple<double, double>> to_predict;
    to_predict.push_back(make_tuple(15.0, 10.0));
    to_predict.push_back(make_tuple(2.0, 2.0));
    to_predict.push_back(make_tuple(7.0, 7.0));
    to_predict.push_back(make_tuple(0.5, 0.6));

    BinarySVM perc;
    ClassificationModelProxy<string, double, double> proxy(perc); // (perc, dataset, targets)
    proxy.fit(dataset, targets);
    auto predicted_values = proxy.predict(to_predict);
    auto predicted_value = proxy.predict(make_tuple(10.0, 5.0));
    cout << "Hello" << endl;
}