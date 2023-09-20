#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

using IDomMapT = llvm::DenseMap<mlir::Value, mlir::Value>;
IDomMapT calcIDom(mlir::func::FuncOp func);
void dumpIDom(const IDomMapT &idomMap);
