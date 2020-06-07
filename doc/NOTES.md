# Notes

## Project goal

The goal of the project is to implement a basic machine learning library in c++. The emphasis is given to interface design rather than actual models implementation. Library should be able to use std type structures as model's input data, namely std::tuple<> as sample/vector representation. Library should also support different std value types such as double, int, std::chrono::time_point, and so on.

## API design choices

Actual models are implemented as classes inheriting from abstract base class `Model` (to be more exact, `Model` class is further divided into abstract `RegressionModel` class and abstract `ClassificationModel` class - these classes are used to be inherited from by concrete models). `Model` class implements few helper methods and data fields used by every model. `RegressionModel` and `ClassificationModel` classes implement fit() and predict() in a specific way (note that these classes use Eigen types to fit data and predict results). They pass actual training and prediction algorithms to their children using virtual methods (fit() and predict() are wrappers for those functions). Each class also have distinct fields and methods used for specific purposes. There are also template proxy classes - `RegressionModelProxy`, `ClassificationModelProxy` - used for conversion automation. User can therefore either wrap their model inside these proxy classes or convert "manually" using methods from "convert_utils.hpp".

Conversion utility methods were divided into 3 static classes:

- `ValueConverter`
  - methods are used for converting between specified types (`Supported` concept)
- `VectorConverter`
  - methods are used for converting between two vector representations: std::vector and Eigen::Matrix<>
- `TupleConverter`
  - methods are used for converting between two vector representations: std::tuple and Eigen::Matrix<>
  - as of today, there is no TupleConverter class and as a temporary solution methods are not inside any class, most probably because of gcc bug "https://gcc.gnu.org/bugzilla//show_bug.cgi?id=79917"
  - this issue will be fixed as soon as possible

Defining supported data types is done via template concepts, using mostly std::is_same<> method.

Metrics measuring methods have both Eigen and std type versions

## Project structure

- The library is implemented in headers located in ./include folder of the project. There is also an external dependency - `Eigen` library. There are total of 6 ".hpp" files:
  - `convert_utils.hpp` - conversion utility automating conversion between std types and Eigen types
  - `metrics.hpp` - model metrics - relevant model measurements
  - `model.hpp` - model implementations
  - their corresponding exception files...
- There are 3 usage examples:
  - basic - located in ./src
  - regression - located in ./regression-example
  - classification - located in ./classification-example
- README.md also serves as complete user documentation
- ./doc folder includes notes, additional documentations,...

## Eigen library

Eigen was chosen basically for the fact that it is the most used library for linear algebra in c++. It has proven to be a very good choice, its versitility and elegant API made it super easy to use.

## Possible library extensions

- data preprocessing utility - for handling null values, cathegorical variables, ...
- more models - logistic regression, decision tree, ...
- statistical data analysis utility - data correlation, ...
- and more...
