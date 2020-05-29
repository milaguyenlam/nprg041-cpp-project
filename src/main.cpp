//#include "../include/model.hpp"

#include "../include/convert_util.hpp"
#include <iostream>
#include <tuple>

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
    auto now = chrono::high_resolution_clock::now();
    auto hours = chrono::time_point_cast<chrono::hours>(now);
    double value = convert_from(hours);

    auto time = convert_to<chrono::time_point<chrono::high_resolution_clock>>(value);
    auto edataset = convert_from_std(dataset);
    auto etargets = convert_from_std(targets);

    map<int, string> converter;
    std::vector<string> string_targets{"ahoj", "cau", "ahoj", "ahoj", "nazdar", "cau"};
    auto estring_targets = convert_from_std(string_targets, converter);

    cout << etargets << endl;
}