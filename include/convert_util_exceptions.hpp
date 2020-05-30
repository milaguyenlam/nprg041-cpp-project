#include <exception>

struct NonEmptyMap : public std::exception
{
    const char *what() const throw()
    {
        return "Given std::map<int, TargetType> has to be empty";
    }
};

struct LabelConversionMissing : public std::exception
{
    const char *what() const throw()
    {
        return "Label conversion missing in given map<int, TargetType>.";
    }
};
