
# Branches and pull requests

- Submit issue for discussion before submitting a pull request
- Always branch from `develop` branch
- Name your branch `<feature name>-dev` for a feature and `<fix name>-fix` for fixing an issue
- Separate with the dash (`-`) character For example:
```console
$ # For a feature
$ git checkout master
$ git checkout -b my-feature-dev
$ # For a fix
$ git checkout master
$ git checkout -b my-bug-fix
```
- Provide extended description in pull request
- Reference the issue in the description
- Keep pull requests focused on single issue
- Whenever possible squash all commits on merge

# Quickstart on CI Platforms

If you are developing HiOp on one of our supported platforms, you may want to
use the dependencies we have already installed there. In that case, you may
source the respective variables script we have configured for that platform:

```console
$ cd /path/to/hiop
$ export MY_CLUSTER=newell # or marianas or ascent
$ source ./scripts/${MY_CLUSTER}Variables.sh
$ mkdir build && cd build
$ cmake ..
$ make -j 16
$ make test
```
Optionally, you may want to modify default CMake configuration with
`ccmake` or `cmake-gui` tools before executing `make` command.

# Reproducing CI Builds

To reproduce the exact build configuration used in our continuous integration
configuration, you may use the following workflow on a supported CI cluster (one
of Ascent, Newell, or Marianas):

```console
$ cd /path/to/hiop
$ export MY_CLUSTER=newell # or marianas or ascent
$ ./BUILD.sh
```

Or if you would like to have all the modules loaded in your interactive session:
```console
$ cd /path/to/hiop
$ export MY_CLUSTER=newell # or marianas or ascent
$ source ./scripts/${MY_CLUSTER}Variables.sh
$ mkdir build && cd build
$ cmake -C ../scripts/defaultCIBuild.cmake ..
$ make -j 16
$ make test
```

Note that this workflow is not encouraged for development, but just reproducing
the build we use for continuous integration.


# Style Guidelines
## Indentation

### Indent with spaces
Use two spaces for indentation, absolutely no tab characters.

Rationale: Different editors may interpret tabs differently. Using spaces
only is more likely to provide consistent look in different editors.

### Macros
Do not indent macros, even when they are nested.
```c++
// Yes:
#ifdef MACRO1
  a = 0;
#ifdef MACRO2
  b = 1;
#endif // MACRO2
#else  // MACRO1
  a = 1;
#endif //MACRO1

// No:
#ifdef MACRO1
  a = 0;
  #ifdef MACRO2
    b = 1;
  #endif // MACRO2
#else  // MACRO1
  a = 1;
#endif //MACRO1
```
### Indentation of access specifiers

Prefer no indentation for `private`, `public`, and `protected` access
specifiers.
```c++
// Yes: public and private keywords aligned with open/close braces 
class Myclass
{
public:
  int my_method();

private:
  int memeber_var_; 
}
```
## Spacing

### Operators

There should be spaces before and after each operator with the
exceptions of:
- Member access operators `.` and `->`.
- Reference `&` and dereference `*` operators.
- Bitwise and logical NOT operators `~` and `!`, respectively.
- Comma separator `,` should have space after, but not before.

### Pointers and references

The character `*` is a part of the pointer type name and there
should be no space between the type name and `*`. Similar for
the character `&` in reference types.

```c++
int* pint;      // Yes!
double& refdbl; // Yes!

int *pint;       // No!
double & refdbl; // No!
```
### Flow control statements

No spaces between `if` (and other control/loop statements) and `(`.
However, do make space before opening brace `{`.
```c++
// Preferred: no space between if and ()
if(some_condition) {
  value += 1;
}

// Discouraged: space in between if and ()
if (some_condition) {
  value += 1;
}

// Discouraged: no space between ) and {.
for(some_condition){
  value += 1;
}
```
### Constructor initialization list
Use two spaces and then `:` followed by space to align the first member initialization. Use only one initialization per line with a trailing comma and align multiple initialization lines with the first declaration (essentially at four spaces).

```c++
MyClass::MyClass()
  : member1_(nullptr),
    other_member2_(0)
{

}
```

Final note on spacing: avoid using trailing spaces, e.g., after member or variable declaration (`int a;//no trailing whitespaces`), after semicolon or braces that end a line in the code, after ifdef/endifs, etc.
## Braces

### Use K&R style

Have the opening brace on the same line as the
statement and the closing brace aligned with the statement.
```c++
// Yes: Use K&R braces
while(a > 0) {
  value += 1;
}

// No: Allman braces
while(a > 0)
{
  value += 1;
}

// No: GNU braces
while(a > 0)
  {
    value += 1;
  }
```

### Position of `else` keyword

Keyword `else` should follow the if closing brace. 

```c++
// Yes: else keyword follows `if` closing brace
if(a > 0) {
  value += 1;
  b = a;
} else {
  value -= 1;
}

// No: else on the next line
if(a > 0) {
  value += 1;
  b = a;
}
else {
  value -= 1;
}
```
### Use braces for every block

For consistency and better code readability use braces
for every code block in selection statements. Also,
avoid condition and loop bodies in the same line of code
```c++
// Yes: Use braces for each block
if(a > 0) {
  value += 1;
  b = a;
} else {
  value -= 1;
}

// No: Use braces only for blocks with more than one statement 
if(a > 0) {
  value += 1;
  b = a;
} else
  value -= 1;

// No: the condition and the body on the same line 
if (a > 0) value += 1;
else       value -= 1;
```


### Do not use one true brace style

For function and class definitions, have the opening brace
on the next line aligned with the definition statement.
```c++
// Yes: opening brace on the next line for functions and classes declarations
bool MyClass::my_method(int a)
{

}

// No: opening brace on the same line
bool MyClass::my_method(int a) {

}
``` 

## RAJA specific style
- Indent two spaces above surrounding scope
- No spaces around template parameter
- Line break before range segment
- Use `RAJA_LAMBDA` define instead of specifying lambda function directly (e.g. `[=] __device__` )
- Use `hiop_*` aliases over specific RAJA policies
- Prefer `RAJA::Index_type` over `int` or anything else
```c++
// RAJA Lambdas
RAJA::forall<hiop_raja_exec>(
  RAJA::RangeSegment(0, 10),
  RAJA_LAMBDA(RAJA::Index_type i)
  {
    if(svec[i] == 1.) {
      local_data_dev[i] = c;
    } else {
      local_data_dev[i] = 0.;
    }
  }
);
```

## Naming Conventions

### Classes

For now prepend `hiop` to the class name. The remaining name should use only upper and lower cases, 
no underscore, i.e., Pascal convention.  Nested classses (class inside a class) should be 
named using  Pascal convention. When the name is composed of multiple words, each word 
starts with a capital letter. If one of the "words" is an acronym use same capitalization 
(first capital and other letters lowercase).
```c++
class hiopMyHiopClass;

class hiopAbc
{
  class Xyz
  {
  };
}
```

### Variables

Use "snake case" for all variables. All lower case; for composite names
separate words with underscores.
```c++
int my_int_variable;
```

### Member variables

All member variable names should end with trailing underscore to distingusish
them from local variables.
```c++
int member_variable_; // Yes!

int another_member_var;      // No!
int _yet_another_member_var; // No!
```
### Function names

Use the snake case for function names, as well.
separate words with underscores.
```c++
void my_hiop_function(int i);
```

Avoid encoding type information in names and use function
overloading instead. For example:
```cpp
// No
void print_double(double val);
void print_str(std::string_view val);

// Yes
void print(double val);
void print(std::string_view val);
```

## Declarations

### Variables

Have each variable declaration have its own type declaration
For example:

```cpp
// Yes
int a;
int b;
int c;

// No
int a, b, c;
```

### Breaking down long lines

Developers should have in mind a limit of ~125 characters per line. 

Here are guidelines for breaking function/method declaration or calls into multiple lines:
- keep the first argument on the same line with the method name
- align arguments on the subsequent lines with the first argument
- if class name or return type long, place the function name at the beginning of the line and put the return type and class name qualification on the previous line (see example below)

```c++
// Yes: all arguments on the same line
void ClassName::some_method(double beta, double alpha) const
{
}

// Yes: each argument on a separate line
void ClassName::some_method(double beta,
                            hiopMatrix& W,
                            double alpha,
                            const hiopMatrix& X) const
{
}

// Yes: when return type + class name + method name + first argument would overrun the 125 characters line limit, have method name on the next line
long_return_type SomeVeryLongClassNameIMeanLong::
some_method(double beta,
            hiopMatrix& W,
            double alpha,
            const hiopMatrix& X) const
{
}

//Yes: function call may need to be broken into multiple line
object1.method1(param1, param2).method2(param).some_method(beta,
                                                           someobject.get_W(),
                                                           alpha,
                                                           X);
                                                           
//Yes: alternative to the above 
object1.method1(param1, param2).method2(param).
  some_method(beta, someobject.get_W(), alpha, X);
                                                           
                                                           
                                                           

// No: multiple arguments per line when using more than one line for the declaration 
void ClassName::some_method(double beta, hiopMatrix& W,
                            double alpha, const hiopMatrix& X) const
{
}

// No: multiple arguments per line when calling
object1.method1(param1).method2(param).some_method(beta, someobject.get_W(),
                                                   alpha, X);
                                                           
// No: arguments not aligned behind the opening parenthesis
void ClassName::some_method(
  double beta,
  hiopMatrix& W,
  double alpha,
  const hiopMatrix& X) 
{
}

// everything above applies to constructors as well, for example,
SomeVeryLongClassNameIMeanLong::
SomeVeryLongClassNameIMeanLong(double beta,
                               hiopMatrix& W,
                               double alpha,
                               const hiopMatrix& X)
  : member1_(0),
    member2_(nullptr)
{
}
```

## Comments
- Don't comment what can be expressed in code
- Comment intent, not implementation
- Block comments (`/* comment */`) are preferred for comments three lines or longer

### Documenting methods

Each method should be documented using Doxygen style comments. At minimum,
there should be description of method arguments, template parameters (if
applicable), return value, pre- and post-conditions. Use template as below.
Input and output arguments should be clearly marked as such (e.g. `@param[in]`
for inputs).

Detailed comments should go in the implementation file (but not prohibited from appearing in the header file). 

Documentation of pre and post conditions should appear in the interface or header file (but not prohibited from appearing in the implementation file). 

```c++
/**
 * @brief <BRIEF DESCRIPTION>
 *
 * @param <NAME> <DESCRIPTION>
 * @tparam <NAME> <DESCRIPTION>
 * @return <DESCRIPTION OF RETURN VALUE>
 *
 * @pre <PRECONDITION>
 * @post <POSTCONDITION>
 *
 * LONGER DESCRIPTION, RUNTIME, EXAMPLES, ETC 
 */
 
```

# Minimal testing procedure 
Any PR or push to the `master` or `develop` branches should go through and pass the procedures below. The shell commands need to be ran in the 'build' directory and assume bash shell. 

## 1. All "build matrix" tests must pass

To run tests for every possible cmake configuration, the build script has an
option for that:

```shell
$> ./BUILD.sh --full-build-matrix
```

If you're going to use this, you'll likely want to create a file in `scripts`
directory with name `./scripts/<cluster>Variables.sh` which sets relevant environment variables, or do this yourself. For example, see `./scripts/ascentVariables.sh` which you would call with:

```shell
$> MY_CLUSTER=ascent ./BUILD.sh --full-build-matrix
```

### Manual testing
To investigate a failure of a test from the build matrix, one can build and run individiual tests from the  build matrix manually, as shown in the examples below.

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
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_EX1.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_EX2.exe 
$> mpiexec -np 2 valgrind ./src/Drivers/nlpDenseCons_EX3.exe 
```

## 3. clang with fsanitize group checks reports no warning and no errors. MacOS only (Optional).
```shell
$> rm -rf *; CC=clang CXX=clang++ cmake -DCMAKE_CXX_FLAGS="-fsanitize=nullability,undefined,integer,alignment" -DHIOP_USE_MPI=ON -DHIOP_DEEPCHECKS=ON -DCMAKE_BUILD_TYPE=DEBUG ..
$> make -j4 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_EX1.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_EX2.exe 
$> mpiexec -np 2 ./src/Drivers/nlpDenseCons_EX3.exe 
```
