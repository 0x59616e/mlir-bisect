#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

void markAsFailed(mlir::Value);
bool isFailed(mlir::Value);

void markAsSuccess(mlir::Value);
bool isSuccess(mlir::Value);

void markAsChecking(mlir::Value);
bool isChecking(mlir::Value);

void markAsCulprit(mlir::Value);
bool isCulprit(mlir::Value);

mlir::Value getCulpritValue(mlir::ModuleOp);
mlir::Value getCheckingValue(mlir::ModuleOp);

bool isFailedOrUnknown(mlir::Value);
bool isUnknown(mlir::Value);

void clearBisectStatus(mlir::ModuleOp mod);