#include "OpMarker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

static const char *statusStr = "status";
static const char *checkingStr = "checking";
static const char *successStr = "success";
static const char *failedStr = "failed";
static const char *culpritStr = "culprit";
static const char *unknownStr = "unknown";

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

static mlir::ArrayAttr getEmptyStatusAttr(mlir::Operation *op) {
  mlir::MLIRContext *ctx = op->getContext();
  int resultNum = op->getNumResults();

  auto str = mlir::StringAttr::get(ctx, unknownStr);
  llvm::SmallVector<mlir::Attribute> values(resultNum, str);

  return mlir::ArrayAttr::get(ctx, values);
}

static mlir::ArrayAttr getNewStatusArray(mlir::ArrayAttr oldStatusArray,
                                         mlir::Attribute attr, int idx) {
  mlir::MLIRContext *ctx = oldStatusArray.getContext();
  auto array = llvm::SmallVector<mlir::Attribute>(oldStatusArray.getValue());
  array[idx] = attr;
  return mlir::ArrayAttr::get(ctx, array);
}

static void markStatus(mlir::Value value, llvm::StringRef status) {
  if (llvm::isa<mlir::BlockArgument>(value))
    return;

  mlir::Operation *op = value.getDefiningOp();
  mlir::MLIRContext *context = op->getContext();
  int resultNum = llvm::cast<mlir::OpResult>(value).getResultNumber();
  mlir::ArrayAttr statusArray;
  assert(op);

  if (op->hasAttr(statusStr))
    statusArray = op->getAttrOfType<mlir::ArrayAttr>(statusStr);
  else
    statusArray = getEmptyStatusAttr(op);

  auto newStatusArray = getNewStatusArray(
      statusArray, mlir::StringAttr::get(context, status), resultNum);

  op->setAttr(statusStr, newStatusArray);
}

static bool checkStatus(mlir::Value value, llvm::StringRef status) {
  if (llvm::isa<mlir::BlockArgument>(value)) {
    return status == successStr;
  }

  mlir::Operation *op = value.getDefiningOp();
  if (llvm::isa<mlir::arith::ConstantOp>(op)) {
    return status == successStr;
  }

  if (!op->hasAttr(statusStr))
    return false;

  int resultNum = llvm::cast<mlir::OpResult>(value).getResultNumber();
  auto statusAttr = op->getAttrOfType<mlir::ArrayAttr>(statusStr)[resultNum];
  if (statusAttr.cast<mlir::StringAttr>().getValue() != status)
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
    for (auto str : {statusStr})
      if (op->hasAttr(str))
        op->removeAttr(str);
  });
}
