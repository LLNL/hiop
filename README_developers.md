# Minimal testing procedure 
Any PR or push to the master should go through and pass the procedures below. The shell commands need to be ran in the 'build' directory and assume bash shell. 

## 1. All tests in 'make test' pass for
1. a. MPI=ON, DEEPCHECKS=OFF, RELEASE=ON (this is the high-performance version of HiOp)
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
1. b. MPI=OFF, DEEPCHECKS=OFF, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
and, optionally, for 
1. c. MPI=ON, DEEPCHECKS=ON, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```
1. d. MPI=OFF, DEEPCHECKS=ON, RELEASE=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=RELEASE ..
$> make -j4; make install; make test
```

1. e. MPI=ON, DEEPCHECKS=OFF, DEBUG=ON 
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4; make install; make test
```

1. f. MPI=OFF, DEEPCHECKS=OFF, DEBUG=ON
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=OFF -DHIOP_DEEPCHECKS=OFF -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4; make install; make test
```

## 2. Valgrind reports no errors and no warning when running the testing drivers of HiOp. Mandatory on Linux, optional on MacOS
```shell
$> rm -rf *; cmake -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex1.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex2.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_ex3.exe 
```

## 3. clang with fsanitize group checks reports no warning and no errors. MacOS only.
```shell
$> rm -rf *; CC=clang CXX=clang++ cmake -DCMAKE_CXX_FLAGS="-fsanitize=nullability,undefined,integer,alignment" -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex1.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex2.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_ex3.exe 
```

# Style Guide

## Contributing

- Submit issue for discussion before submitting a pull request
- Always branch from master
- Name your branch `<feature name>-dev` for a feature and `<fix name>-fix` for fixing an issue
- Separate with the dash (`-`) character
- Provide extended description in pull request
- Reference the issue in the description
- Squash all commits on merge

For example:
```console
$ # For a feature
$ git checkout master
$ git checkout -b my-feature-dev
$ # For a fix
$ git checkout master
$ git checkout -b my-bug-fix
```

## Indentation, Braces, and Declarations

- Use two spaces for indentation, absolutely no tab characters.
- No spaces between `if` and `(`
- Avoid condition and loop bodies in the same line of code
- Allman style braces
- RAJA Lambdas
  - Indent two spaces above surrounding scope
  - No spaces around template parameter
  - Line break before range segment
  - Use `RAJA_LAMBDA` in favor of using `__device__` directly
  - Use `hiop_*_policy` over directly using cuda or omp policies
  - Prefer `RAJA::Index_type` over `int` or anything else
- Prefer braces for every block
- Prefer no indentation for `private`, `public`, and `protected` statements
- Each variable declaration should have it's own type declaration

For example:

```cpp
// Good
int a;
int b;
int c;

// Bad
// this causes confusion when working with pointers, and if you
// must change a type in the future, we will have a larger diff.
int a, b, c;

// Good
// uses allman braces
// no space between if and ()
if(some_condition)
{
  value += 1;
}

// Not preferred
// use allman braces
if(some_condition) {
  value += 1;
}

// Not preferred
// space in between if and ()
if (some_condition)
{
  value += 1;
}

// Not preferred
// please use braces
if(some_condition)
  value += 1;

// Bad!
if(some_condition) value += 1;

// RAJA Lambdas
RAJA::forall<hiop_raja_exec>(
  RAJA::RangeSegment(0, 10),
  RAJA_LAMBDA(RAJA::Index_type i)
  {
    if(svec[i]==1.)
    {
      local_data_dev[i] = c;
    }
    else
    {
      local_data_dev[i]=0.;
    }
  });

```

## Naming Conventions

- Classes
  - Pascal case (upper and lower case, no underscores)
- Data members
  - Snake case
    - lower case only
    - use _ to separate
  - Ending in _
- Member methods
  - Snake case
    - lower case only
    - use _ to separate
- Avoid encoding type information in names

For example:
```cpp
// Bad, prefer polymorphism
void print_double(double val);
void print_str(std::string_view val);

// Good
void print(double val);
void print(std::string_view val);

// Good
class FooBar
{
private:
  double* data_device_;
public:
  void to_device();
}

// Bad
class Foo_bar {
  private:
    double* dataDevice;
  public:
    void toDevice();
}
```

## Comments

- Don't comment what can be expressed in code
- Comment intent, not implementation
- Block comments (`/* comment */`) are preferred for comments three lines or longer
