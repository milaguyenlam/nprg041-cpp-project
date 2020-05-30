#include <exception>

struct InvalidTargetFormat : public std::exception
{
    const char *what() const throw()
    {
        return "Classification labels has to be in following format: int (0, 1, ... , N) when having N classification classes.";
    }
};

struct BinaryLabelsExpected : public std::exception
{
    const char *what() const throw()
    {
        return "Model only supports binary classification. More than 2 classes detected.";
    }
};

struct PredictBeforeFit : public std::exception
{
    const char *what() const throw()
    {
        return "Model has to be trained (fit() method) before predicting.";
    }
};

struct InvalidTrainingDataFormat : public std::exception
{
    const char *what() const throw()
    {
        return "Number of samples doesn't equal number of corresponding targets.";
    }
};

struct InvalidPredictDataFormat : public std::exception
{
    const char *what() const throw()
    {
        return "Predict data dimension (number of sample features) doesn't correspond to training data's";
    }
};