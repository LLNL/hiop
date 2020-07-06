maintaining features and issues, mostly from Shri

* C wrapper for hiop mds interface
* The equality and inequality constraints need to be a function of at least one sparse variable. 
* HIOP currently does not support  no inequality constraints 
* HIOP currently does not support  no sparse variables
* hiop calls the constraints routine twice -> Cosmin note: this turns out to be quite inconviniently
    * an ipopt-like "one Jacobian call" can be implemented with insignificant performance hit
* The sparse triplet format used as input by HiOp requires
    * all the i and j pairs should be unique
    * the entries should be ordered first by row and then by column
    * Cosmin note: low priority for now
* Missing headers hiopMatrixMDS.hpp and others
    * this is fixed now