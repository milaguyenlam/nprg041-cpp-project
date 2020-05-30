#include <chrono>
#include <vector>
#include <map>
#include "Eigen/Core"

//TODO: add include guards
//TODO: add array conversions

template <typename T>
concept SupportedClassificationTargets = std::is_same<T, int>::value || std::is_same<T, std::string>::value || std::is_same<T, char>::value || std::is_same<T, long>::value;

template <typename T>
concept SupportedTimePoints =
    std::is_same_v<T, std::chrono::time_point<typename T::clock, typename T::duration>>;

template <typename T>
concept Supported =
    std::is_same<T, int>::value || std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, long>::value || SupportedTimePoints<T>;

class SingleValueConverter
{
public:
    template <Supported From>
    static double convert_from(From value)
    {
        return (double)value;
    }

    template <class Clock, class Duration>
    static double convert_from(const std::chrono::time_point<Clock, Duration> &value)
    {
        return (double)value.time_since_epoch().count();
    }

    template <Supported To>
    static To convert_to(double value)
    {
        if constexpr (SupportedTimePoints<To>)
        {
            using duration = typename To::duration;
            duration dur((long)value);
            To time(dur);
            return time;
        }
        else
        {
            return (To)value;
        }
    }

private:
    SingleValueConverter()
    {
    }
};
//TODO: convert to vector<vector<T>>
class VectorConverter
{
public:
    template <Supported From>
    static Eigen::VectorXd convert_from_std(const std::vector<From> &vector)
    {
        size_t vector_length = vector.size();
        Eigen::VectorXd evector(vector_length);
        for (size_t i = 0; i < vector_length; i++)
        {
            auto converted_value = SingleValueConverter::convert_from(vector[i]);
            evector[i] = converted_value;
        }
        return evector;
    }

    template <Supported To>
    static std::vector<To> convert_to_std(const Eigen::VectorXd &vector)
    {
        std::vector<To> svector;
        size_t vector_lenght = vector.rows();
        for (size_t i = 0; i < vector_lenght; i++)
        {
            To value = SingleValueConverter::convert_to<To>(vector[i]);
            svector.push_back(value);
        }
        return svector;
    }

    template <Supported From>
    static Eigen::MatrixXd convert_from_std(const std::vector<std::vector<From>> &matrix)
    {
        size_t columns = matrix.size();
        size_t rows = matrix[0].size(); //should be checked for every element
        Eigen::MatrixXd m(rows, columns);
        for (size_t i = 0; i < columns; i++)
        {
            auto vector = matrix[i];
            for (size_t j = 0; j < rows; j++)
            {
                auto converted_value = SingleValueConverter::convert_from(vector[j]);
                m(j, i) = converted_value;
            }
        }
        return m;
    }

    template <SupportedClassificationTargets TargetType>
    static Eigen::VectorXi convert_from_std(const std::vector<TargetType> &vector, std::map<int, TargetType> &converter)
    {
        //TODO: test if map is empty
        std::map<TargetType, int> helper_map;
        std::size_t length = vector.size();
        Eigen::VectorXi ret_vector(length);
        int counter = 0;
        for (std::size_t i = 0; i < length; i++)
        {
            auto target = vector[i];
            if (helper_map.count(target) > 0)
            {
                int key = helper_map.find(target)->second;
                ret_vector[i] = key;
            }
            else
            {
                helper_map.insert(std::pair<TargetType, int>(target, counter));
                converter.insert(std::pair<int, TargetType>(counter, target));
                ret_vector[i] = counter;
                counter++;
            }
        }
        return ret_vector;
    }

    template <SupportedClassificationTargets TargetType>
    static std::vector<TargetType> convert_to_std(const Eigen::VectorXi &vector, const std::map<int, TargetType> &converter)
    {
        std::vector<TargetType> ret_values;
        std::size_t length = vector.size();
        for (size_t i = 0; i < length; i++)
        {
            TargetType ret_value = converter.find(vector[i])->second;
            ret_values.push_back(ret_value);
        }
        return ret_values;
    }

private:
    VectorConverter()
    {
    }
};

class ArrayVectorConverter
{
private:
    ArrayVectorConverter()
    {
    }
};

template <class Type>
void push_value_to_vector(std::vector<double> &vector, Type item)
{
    vector.push_back(SingleValueConverter::convert_to<double>(item));
}

template <size_t I, class... Types>
void iterate_recursively_and_push_value(std::vector<double> &vector, const std::tuple<Types...> &tuple)
{
    push_value_to_vector(vector, std::get<I>(tuple));

    if constexpr (I + 1 < sizeof...(Types))
    {
        return iterate_recursively_and_push_value<I + 1, Types...>(vector, tuple);
    }
}

template <typename... Types>
void push_converted_data_to_vector(std::vector<double> &vector, const std::tuple<Types...> &tuple)
{
    return iterate_recursively_and_push_value<0, Types...>(vector, tuple);
}

template <Supported... From>
Eigen::VectorXd convert_from_std(const std::tuple<From...> &tuple)
{
    std::vector<double> vector;
    push_converted_data_to_vector(vector, tuple);
    Eigen::VectorXd evector = VectorConverter::convert_from_std<double>(vector);
    return evector;
}

template <Supported... From>
Eigen::MatrixXd convert_from_std(const std::vector<std::tuple<From...>> &tuples)
{
    std::size_t matrix_cols = tuples.size();
    std::size_t matrix_rows = sizeof...(From);
    Eigen::MatrixXd ret_matrix(matrix_rows, matrix_cols);

    for (std::size_t i = 0; i < matrix_cols; i++)
    {
        Eigen::VectorXd vector = convert_from_std<From...>(tuples[i]);
        for (std::size_t j = 0; j < matrix_rows; j++)
        {
            double value = vector[j];
            ret_matrix(j, i) = value;
        }
    }
    return ret_matrix;
}

//TODO: convert to tuple and vector<tuple>
// class TupleConverter
// {
// public:
//     template <Supported... From>
//     static Eigen::VectorXd convert_from_std(const std::tuple<From...> &tuple)
//     {
//         std::vector<double> vector;
//         push_converted_data_to_vector(vector, tuple);
//         Eigen::VectorXd evector = VectorConverter::convert_from_std<double>(vector);
//         return evector;
//     }

//     template <Supported... From>
//     static Eigen::MatrixXd convert_from_std(const std::vector<std::tuple<From...>> &tuples)
//     {
//         std::size_t matrix_cols = tuples.size();
//         std::size_t matrix_rows = sizeof...(From);
//         Eigen::MatrixXd ret_matrix(matrix_rows, matrix_cols);

//         for (std::size_t i = 0; i < matrix_cols; i++)
//         {
//             Eigen::VectorXd vector = TupleConverter::convert_from_std<From...>(tuples[i]);
//             for (std::size_t j = 0; j < matrix_rows; j++)
//             {
//                 double value = vector[j];
//                 ret_matrix(j, i) = value;
//             }
//         }
//         return ret_matrix;
//     }

// private:
//     TupleConverter()
//     {
//     }

//     template <typename... Types>
//     static void push_converted_data_to_vector(std::vector<double> &vector, const std::tuple<Types...> &tuple)
//     {
//         return iterate_recursively_and_push_value<0, Types...>(vector, tuple);
//     }

//     template <size_t I, class... Types>
//     static void iterate_recursively_and_push_value(std::vector<double> &vector, const std::tuple<Types...> &tuple)
//     {
//         push_value_to_vector(vector, std::get<I>(tuple));

//         if constexpr (I + 1 < sizeof...(Types))
//         {
//             return iterate_recursively_and_push_value<I + 1, Types...>(vector, tuple);
//         }
//     }
//     template <class Type>
//     static void push_value_to_vector(std::vector<double> &vector, Type item)
//     {
//         vector.push_back(convert_to<double>(item));
//     }
// };

//is overriden(time_point case), should be called only for long, double, int, float
// template <Supported From>
// double convert_from(From value)
// {
//     return (double)value;
// }

// template <class Clock, class Duration>
// double convert_from(const std::chrono::time_point<Clock, Duration> &value)
// {
//     return (double)value.time_since_epoch().count();
// }

// template <Supported To>
// To convert_to(double value)
// {
//     if constexpr (SupportedTimePoints<To>)
//     {
//         using clock = typename To::clock;
//         using duration = typename To::duration;
//         duration dur((long)value);
//         To time(dur);
//         return time;
//     }
//     else
//     {
//         return (To)value;
//     }
// }

// template <Supported From>
// Eigen::VectorXd convert_from_std(const std::vector<From> &vector)
// {
//     size_t vector_length = vector.size();
//     Eigen::VectorXd evector(vector_length);
//     for (size_t i = 0; i < vector_length; i++)
//     {
//         evector[i] = vector[i];
//     }
//     return evector;
// }

// template <Supported To>
// std::vector<To> convert_to_std(const Eigen::VectorXd &vector)
// {
//     std::vector<To> svector;
//     size_t vector_lenght = vector.rows();
//     for (size_t i = 0; i < vector_lenght; i++)
//     {
//         To value = convert_to<To>(vector[i]);
//         svector.push_back(value);
//     }
//     return svector;
// }

// template <Supported... From>
// Eigen::VectorXd convert_from_std(const std::tuple<From...> &tuple)
// {
//     std::vector<double> vector;
//     push_converted_data_to_vector(vector, tuple);
//     Eigen::VectorXd evector = convert_from_std<double>(vector);
//     return evector;
// }

// template <Supported From>
// Eigen::MatrixXd convert_from_std(const std::vector<std::vector<From>> &matrix)
// {
//     size_t columns = matrix.size();
//     size_t rows = matrix[0].size(); //should be checked for every element
//     Eigen::MatrixXd m(rows, columns);
//     for (size_t i = 0; i < columns; i++)
//     {
//         auto vector = matrix[i];
//         for (size_t j = 0; j < rows; j++)
//         {
//             m(j, i) = vector[j];
//         }
//     }
//     return m;
// }

// template <Supported... From>
// Eigen::MatrixXd convert_from_std(const std::vector<std::tuple<From...>> &tuples)
// {
//     std::size_t matrix_cols = tuples.size();
//     std::size_t matrix_rows = sizeof...(From);
//     Eigen::MatrixXd ret_matrix(matrix_rows, matrix_cols);

//     for (std::size_t i = 0; i < matrix_cols; i++)
//     {
//         Eigen::VectorXd vector = convert_from_std<From...>(tuples[i]);
//         for (std::size_t j = 0; j < matrix_rows; j++)
//         {
//             double value = vector[j];
//             ret_matrix(j, i) = value;
//         }
//     }
//     return ret_matrix;
// }

// template <SupportedClassificationTargets TargetType>
// Eigen::VectorXi convert_from_std(const std::vector<TargetType> &vector, std::map<int, TargetType> &converter)
// {
//     std::map<TargetType, int> helper_map;
//     std::size_t length = vector.size();
//     Eigen::VectorXi ret_vector(length);
//     int counter = 0;
//     for (std::size_t i = 0; i < length; i++)
//     {
//         auto target = vector[i];
//         if (helper_map.count(target) > 0)
//         {
//             int key = helper_map.find(target)->second;
//             ret_vector[i] = key;
//         }
//         else
//         {
//             helper_map.insert(std::pair<TargetType, int>(target, counter));
//             converter.insert(std::pair<int, TargetType>(counter, target));
//             ret_vector[i] = counter;
//             counter++;
//         }
//     }
//     return ret_vector;
// }

// template <SupportedClassificationTargets TargetType>
// std::vector<TargetType> convert_to_std(const Eigen::VectorXi &vector, const std::map<int, TargetType> &converter)
// {
//     std::vector<TargetType> ret_values;
//     std::size_t length = vector.size();
//     for (size_t i = 0; i < length; i++)
//     {
//         TargetType ret_value = converter.find(vector[i])->second;
//         ret_values.push_back(ret_value);
//     }
//     return ret_values;
// }