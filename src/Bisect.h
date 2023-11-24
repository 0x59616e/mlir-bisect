#pragma once
#include "mlir/IR/BuiltinOps.h"

enum class SearchStatus {
  YetToBeFound,
  FoundSuccessfully,
};

SearchStatus searchCulprit(mlir::ModuleOp);
