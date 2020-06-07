#include <exception>

struct InvalidDataFormat : public std::exception
{
    const char *what() const throw()
    {
        return "Number of predictions doesn't equal number of targets";
    }
};