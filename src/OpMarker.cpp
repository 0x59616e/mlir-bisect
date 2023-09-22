#include "OpMarker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

static const char *statusStr = "status";
static const char *checkingStr = "checking";
static const char *successStr = "success";
static const char *failedStr = "failed";
static const char *culpritStr = "culprit";
static const char *whichStr = "which";

mlir::Value getCheckingValue(mlir::ModuleOp mod) {
  mlir::Value theValue;
  mod.walk([&](mlir::Operation *op) {
    for (auto result : op->getResults())
      if (isChecking(result))
        theValue = result;
  });

  return theValue;
}

mlir::Value getCulpritValue(mlir::ModuleOp mod) {
  mlir::Value theValue;
  mod.walk([&](mlir::Operation *op) {
    for (auto result : op->getResults())
      if (isCulprit(result))
        theValue = result;
  });

  return theValue;
}

static void markStatus(mlir::Value value, llvm::StringRef status) {
  if (llvm::isa<mlir::BlockArgument>(value))
    return;

  mlir::Operation *op = value.getDefiningOp();
  mlir::MLIRContext *context = op->getContext();
  int which = llvm::cast<mlir::OpResult>(value).getResultNumber();
  assert(op);
  auto statusAttr = mlir::StringAttr::get(context, status);
  auto whichAttr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), which);

  op->setAttr(statusStr, statusAttr);
  op->setAttr(whichStr, whichAttr);
}

static bool checkStatus(mlir::Value value, llvm::StringRef status) {
  if (llvm::isa<mlir::BlockArgument>(value)) {
    return status == successStr;
  }

  mlir::Operation *op = value.getDefiningOp();
  if (llvm::isa<mlir::arith::ConstantOp>(op)) {
    return status == successStr;
  }

  int resultNum = llvm::cast<mlir::OpResult>(value).getResultNumber();

  auto statusAttr = op->getAttr(statusStr);
  if (!statusAttr)
    return false;

  if (statusAttr.cast<mlir::StringAttr>().getValue() != status)
    return false;

  auto whichAttr = op->getAttr(whichStr);
  assert(whichAttr);
  if (resultNum !=
      whichAttr.cast<mlir::IntegerAttr>().getValue().getZExtValue())
    return false;

  return true;
}

void markAsFailed(mlir::Value value) {
  if (isFailed(value))
    return;

  markStatus(value, failedStr);

  for (auto user : value.getUsers()) {
    for (auto userResult : user->getResults())
      markAsFailed(userResult);
  }
}
bool isFailed(mlir::Value value) { return checkStatus(value, failedStr); }

void markAsSuccess(mlir::Value value) {
  if (isSuccess(value))
    return;

  markStatus(value, successStr);

  mlir::Operation *op = value.getDefiningOp();
  if (!op)
    return;

  for (auto operand : op->getOperands())
    markAsSuccess(operand);
}
bool isSuccess(mlir::Value value) { return checkStatus(value, successStr); }

bool isFailedOrUnknown(mlir::Value value) {
  return isFailed(value) || !isSuccess(value);
}
bool isUnknown(mlir::Value value) {
  return !isFailed(value) && !isSuccess(value);
}

void markAsChecking(mlir::Value value) { markStatus(value, checkingStr); }
bool isChecking(mlir::Value value) { return checkStatus(value, checkingStr); }

void markAsCulprit(mlir::Value value) { markStatus(value, culpritStr); }
bool isCulprit(mlir::Value value) { return checkStatus(value, culpritStr); }

void clearBisectStatus(mlir::ModuleOp mod) {
  mod.walk([&](mlir::Operation *op) {
    for (auto str : {statusStr, whichStr})
      if (op->hasAttr(str))
        op->removeAttr(str);
  });
}