# xSDK Community Policy Compatibility for HiOp

This document summarizes the efforts of current and future xSDK member packages to achieve compatibility with the xSDK community policies. Below only short descriptions of each policy are provided. The full description is available [here](https://github.com/xsdk-project/xsdk-community-policies)
and should be considered when filling out this form.

Please, provide information on your compability status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.

**Website:** https://github.com/LLNL/hiop

### Mandatory Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support portable installation through Spack. |Full| Private fully-functional Spack packages in the process of being merged upstream.|
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| HiOp contains several test drivers for integration tests as well as about 240 unit tests for each key linear algebra kernel.|
|**M3.** Employ user-provided MPI communicator (no MPI_COMM_WORLD). Don't assume a full MPI 3 implementation without checking. Provide an option to prevent any changes to MPI error-handling if it is changed by default. |Full| HiOp objects take an MPI communicator in the constructor. |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| Our continuous integration tests every branch on multiple platforms. |
|**M5.** Provide a documented, reliable way to contact the development team. |Full| [Submit issues here](https://github.com/LLNL/hiop/issues). |
|**M6.** Respect system resources and settings made by other previously called packages (e.g. signal handling). |Full| None. |
|**M7.** Come with an open source (BSD style) license. |Full| Use 2-clause BSD license. |
|**M8.** Provide a runtime API to return the current version number of the software. |Full| Header `hiopVersion.hpp` provides functions and data such as `void version(int& major, int& minor, int& patch)`. |
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| None. |
|**M10.** Provide an xSDK team accessible repository (not necessarily publicly available). |Full| None. |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Full| HiOp takes options to toggle verbosity level. |
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| Our CMake configuration allows for linking against externally built packages. |
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full| None. |
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full| None. |
|**M15.** All xSDK compatibility changes should be sustainable. |Full| None. |
|**M16.** Any xSDK-compatible package that compiles code should have a configuration option to build in Debug mode. |Full| None. |

M1 details <a id="m1-details"></a>: optional: provide more details about approach to addressing topic M1.

M2 details <a id="m2-details"></a>: optional: provide more details about approach to addressing topic M2.

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Full| [Public repository on GitHub](https://github.com/LLNL/hiop/). |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |Full| None. |
|**R3.** Adopt and document consistent system for error conditions/exceptions. |Full| Linear algebra kernels use return codes for error handling. |
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Partial| Some destructors free memory if this has not already been handled, indicating some resources may be under program acquisition longer than needed. |
|**R5.** Provide a mechanism to export ordered list of library dependencies. |Full| Header `hiopVersion.hpp` provides dynamic and static dependency information. |
|**R6.** Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |Partial| Spack packages provide this information. |
|**R7.** Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.|Partial| README.md, COPYRIGHT and LICENSE files exist in top-level directory. |
|**R8.** Each xSDK member package should have sufficient documentation to support use and further development.  |Full| None. |
