# xSDK Community Policy Compatibility for \<package\>

This document summarizes the efforts of current and future xSDK member packages to achieve compatibility with the xSDK community policies. Below only short descriptions of each policy are provided. The full description is available [here](https://github.com/xsdk-project/xsdk-community-policies)
and should be considered when filling out this form.

*** A good example of how to complete this form can be found in the [PETSc version](https://github.com/xsdk-project/xsdk-policy-compatibility/blob/master/petsc-policy-compatibility.md).

Please, provide information on your compability status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.

For current xSDK member packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future xSDK member packages.

**Website:** \<https://github.com/LLNL/hiop\>

### Mandatory Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support portable installation through Spack. |Full| Working spack packages thanks to ORNL collaborators (flush out)
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| HiOp contains several test drivers for integration tests as well as >100 unit tests for each key linear algebra kernel.
|**M3.** Employ user-provided MPI communicator (no MPI_COMM_WORLD). Don't assume a full MPI 3 implementation without checking. Provide an option to prevent any changes to MPI error-handling if it is changed by default. |Full| HiOp objects take an MPI communicator in the constructor. |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| Our continuous integration tests every branch on multiple platforms. |
|**M5.** Provide a documented, reliable way to contact the development team. |None| None. |
|**M6.** Respect system resources and settings made by other previously called packages (e.g. signal handling). |Full| None. |
|**M7.** Come with an open source (BSD style) license. |Full| Use 2-clause BSD license. |
|**M8.** Provide a runtime API to return the current version number of the software. |Full| None. |
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| None. |
|**M10.** Provide an xSDK team accessible repository (not necessarily publicly available). |Full| None. |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Full| None. |
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| None. |
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full| None. |
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full| Packages supports both 32 and 64 bit under same API. |
|**M15.** All xSDK compatibility changes should be sustainable. |Full| None. |
|**M16.** Any xSDK-compatible package that compiles code should have a configuration option to build in Debug mode. |Full| None. |

M1 details <a id="m1-details"></a>: optional: provide more details about approach to addressing topic M1.

M2 details <a id="m2-details"></a>: optional: provide more details about approach to addressing topic M2.

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Partial| Repo is access controlled but available as release tarfile. |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |Full| None. |
|**R3.** Adopt and document consistent system for error conditions/exceptions. |None| None. |
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Full| None. |
|**R5.** Provide a mechanism to export ordered list of library dependencies. |None| None. |
|**R6.** Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |None| None. |
|**R7.** Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.  |None| None. |
|**R8.** Each xSDK member package should have sufficient documentation to support use and further development.  |None| None. |
