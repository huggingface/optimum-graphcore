// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::string getSDKVersionHash() {
  // Replace parenthesis and space in version string so
  // we can easily use the results as a variable in a
  // Makefile and on the compiler command line:
  std::string version = poplar::packageHash();
  for (char &c : version) {
    if (c == '(' || c == ')' || c == ' ') {
      c = '_';
    }
  }
  return version;
}

PYBIND11_MODULE(sdk_version_hash_lib, m) {
  m.def("sdk_version_hash", &getSDKVersionHash,
        "Graphcore SDK version hash (sanitised `poplar::packageHash`)");
};

// -------------- cppimport --------------
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-Wall', '-Wsign-compare']
cfg['libraries'] = ['poplar']
setup_pybind11(cfg)
%>
*/