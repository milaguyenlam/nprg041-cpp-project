#include <chrono>
#include <vector>
#include <map>
#include "Eigen/Core"
#include "convert_util_exceptions.hpp"

//TODO: add include guards

namespace convert
{
    //types that can be used as classification labels
    template <typename T>
    concept SupportedClassificationTargets = std::is_same<T, int>::value || std::is_same<T, std::string>::value || std::is_same<T, char>::value || std::is_same<T, long>::value;

    //helper concept for encapsulating std::chrono::time_point class
    template <typename T>
    concept SupportedTimePoints =
        std::is_same_v<T, std::chrono::time_point<typename T::clock, typename T::duration>>;

    //types that are supported for conversion (Supported <-> double)
    template <typename T>
    concept Supported =
        std::is_same<T, int>::value || std::is_same<T, double>::value || std::is_same<T, float>::value || std::is_same<T, long>::value || SupportedTimePoints<T>;

    //static class used for converting between double and supported types (both directions)
    class SingleValueConverter
    {
    public:
        template <Supported From>
        static double convert_from(From value)
        {
            return (double)value;
        }

        //converting from time_point class
        template <class Clock, class Duration>
        static double convert_from(const std::chrono::time_point<Clock, Duration> &value)
        {
            return (double)value.time_since_epoch().count();
        }

        template <Supported To>
        static To convert_to(double value)
        {
            //detecting if "To" type is a time_point, using concept as a "boolean"
            if constexpr (SupportedTimePoints<To>)
            {
                using duration = typename To::duration;
                duration dur((long)value); //instantiating given std::chrono::duration class
                To time(dur);              //instantiating time_point with duration
                return time;
            }
            else
            {
                return (To)value;
            }
        }

    private:
        //constructor as a private method making the class static
        SingleValueConverter()
        {
        }
    };

    //class used for converting std::vector as a vector from/to Eigen data types
    //uses From, To of Supported types as type paramaters
    class VectorConverter
    {
    public:
        //overload for a single vector
        template <Supported From>
        static Eigen::VectorXd convert_from_std(const std::vector<From> &vector)
        {
            //iterating through every value in std::vector and converting it to "From" type
            size_t vector_length = vector.size();
            Eigen::VectorXd evector(vector_length);
            for (size_t i = 0; i < vector_length; i++)
            {
                auto converted_value = SingleValueConverter::convert_from(vector[i]);
                evector[i] = converted_value;
            }
            return evector;
        }
        //overload for a matrix represented as nested std::vector<std::vector>
        template <Supported From>
        static Eigen::MatrixXd convert_from_std(const std::vector<std::vector<From>> &matrix)
        {
            //iterating through every value in std::vector and converting it to "From" type
            size_t columns = matrix.size();
            size_t rows = matrix[0].size();
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
        //converting Eigen vector type to std::vector of "To" type
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

        //function used for converting std::vector to Eigen vector type (integer)
        //maps contents as classification labels to {0,1,...,N}, where N is number of values inside given vector
        //also fills a std::map with "back direction" conversions, given map has to be empty
        template <SupportedClassificationTargets TargetType>
        static Eigen::VectorXi convert_from_std(const std::vector<TargetType> &vector, std::map<int, TargetType> &converter)
        {
            if (!converter.empty())
            {
                throw NonEmptyMap();
            }
            std::map<TargetType, int> helper_map; //map that
            std::size_t length = vector.size();
            Eigen::VectorXi ret_vector(length);
            int counter = 0;
            for (std::size_t i = 0; i < length; i++)
            {
                //iterating through every input vector value
                auto target = vector[i];
                if (helper_map.count(target) > 0)
                {
                    //checking if current target has already occured
                    //if so, save assigned int value into output vector (at i-th index)
                    int value = helper_map.find(target)->second;
                    ret_vector[i] = value;
                }
                else
                {
                    //insert conversion into both maps and save assigned int value
                    helper_map.insert(std::pair<TargetType, int>(target, counter));
                    converter.insert(std::pair<int, TargetType>(counter, target));
                    ret_vector[i] = counter;
                    counter++;
                }
            }
            return ret_vector;
        }

        //used for converting classification model output type (Eigen int vector) to std::vector<TargetType> using map filled with int->TargetType conversions
        template <SupportedClassificationTargets TargetType>
        static std::vector<TargetType> convert_to_std(const Eigen::VectorXi &vector, const std::map<int, TargetType> &converter)
        {
            std::vector<TargetType> ret_values;
            std::size_t length = vector.size();
            for (size_t i = 0; i < length; i++)
            {
                if (!converter.contains(vector[i])) //check if conversion exists in given map
                {
                    throw LabelConversionMissing();
                }
                //save converted value into output vector
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

} // namespace convert

namespace details
{
    using namespace convert;

    //converts data to specified Type and pushes adds it to given vector
    template <class Type>
    void push_value_to_vector(std::vector<double> &vector, Type item)
    {
        vector.push_back(SingleValueConverter::convert_to<double>(item));
    }

    //iterating through std::tuple using static recursion
    //I - specifies index that should be used to be added to vector
    template <size_t I, class... Types>
    void iterate_recursively_and_push_value(std::vector<double> &vector, const std::tuple<Types...> &tuple)
    {
        push_value_to_vector(vector, std::get<I>(tuple)); //adding current tuple value (at index I)

        if constexpr (I + 1 < sizeof...(Types)) //checking if there's next value in tuple
        {
            return iterate_recursively_and_push_value<I + 1, Types...>(vector, tuple); //calling itself to add next value
        }
    }
    //wrapper method for actual recursion
    template <typename... Types>
    void push_converted_data_to_vector(std::vector<double> &vector, const std::tuple<Types...> &tuple)
    {
        return iterate_recursively_and_push_value<0, Types...>(vector, tuple);
    }
} // namespace details

namespace convert
{
    using namespace details;
    //methods for std::tuple conversion
    //putting them into a class did not work, see https://gcc.gnu.org/bugzilla//show_bug.cgi?id=79917
    //they are left as global methods as a temporary solution

    //converting from std::tuple to Eigen vector class
    template <Supported... From>
    Eigen::VectorXd convert_from_std(const std::tuple<From...> &tuple)
    {
        //using recursive method to iterate through std::tuple to fill std::vector
        std::vector<double> vector;
        push_converted_data_to_vector(vector, tuple); //recursive helper method
        Eigen::VectorXd evector = VectorConverter::convert_from_std<double>(vector);
        return evector;
    }
    //overload for matrix represented as std::vector<tuple<>>
    template <Supported... From>
    Eigen::MatrixXd convert_from_std(const std::vector<std::tuple<From...>> &tuples)
    {
        //iterating through every tuple in vector and calling an overload for a single tuple
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
} // namespace convert
