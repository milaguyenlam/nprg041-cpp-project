#include <vector>
#include <tuple>
using namespace std;

//conversion utility
double convertToDouble(string value); //using one hot encoding?
double convertToDouble(int value);
double convertToDouble(double value);
//... version for every supported type

template <typename T>
T convertFromDouble(double value);

template <typename T>
class Model
{
protected:
    vector<double> weights;
    virtual vector<double> fit_algorithm(const vector<vector<double>> &data, const vector<double> &targets);

public:
    template <typename... Types>
    void fit(const vector<tuple<Types...>> &training_data, const vector<T> &training_targets);

    T predict(const vector<double> &vector);
    vector<T> predict(const vector<vector<double>> &vectors);
};

template <typename T>
class LinearRegression : public Model
{
protected:
    vector<double> fit_algorithm(const vector<vector<double>> &data, const vector<double> &targets) override;
};

template <typename T>
class LogisticRegression : public Model
{
protected:
    vector<double> fit_algorithm(const vector<vector<double>> &data, const vector<double> &targets) override;
};