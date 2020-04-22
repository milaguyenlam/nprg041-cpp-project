#MLlib - api design

//where T is Int, Double, Timestamp, Date,... - basically everything convertible to and from double type
//where Types are types convertible to double type

class Model<T> {
public:
  virtual void fit(vector<tuple<Types...>> data, vector<T> targets);
  T predict(tuple<Types...> vector);
  vector<T> predict(vector<tuple<Types...>> data);
protected:
  vector<Double> weights;
}

class LinearRegression<T> : Model<T> {
  void fit(...);
}

class LogisticRegression<T> : Model<T> {
  void fit(...);
}

//Type conversion??? 
- method convert_to(), convert_from()
- implement different Model<Int>, Model<Time>,...
- typeof() swtich/case
