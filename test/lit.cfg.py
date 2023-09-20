# -*- Python -*-

import os
import lit
import lit.formats

config.name = "mlir-bisect"

config.test_format = lit.formats.ShTest()

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.cmake_binary_dir, "test")

config.substitutions.append(("%mlir-bisect", os.path.join(config.cmake_binary_dir, "src", "mlir-bisect")))
config.substitutions.append(("%FileCheck", os.path.join(config.llvm_binary_dir, "bin", "FileCheck")))
