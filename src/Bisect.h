#pragma once

#include "DomTree.h"
#include "mlir/IR/BuiltinOps.h"

enum class SearchStatus {
  YetToBeFound,
  FoundSuccessfully,
};

SearchStatus searchCulprit(mlir::ModuleOp, const IDomMapT &);
