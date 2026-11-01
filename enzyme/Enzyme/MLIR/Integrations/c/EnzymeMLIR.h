#ifndef ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
#define ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Probabilistic Programming Ops
//===----------------------------------------------------------------------===//

typedef enum {
  EnzymeRngDistribution_Uniform = 0,
  EnzymeRngDistribution_Normal = 1,
  EnzymeRngDistribution_MultiNormal = 2,
} EnzymeRngDistribution;

MLIR_CAPI_EXPORTED MlirAttribute
enzymeRngDistributionAttrGet(MlirContext ctx, EnzymeRngDistribution dist);

typedef enum {
  EnzymeSupportKind_Real = 0,
  EnzymeSupportKind_Positive = 1,
  EnzymeSupportKind_UnitInterval = 2,
  EnzymeSupportKind_Interval = 3,
  EnzymeSupportKind_GreaterThan = 4,
  EnzymeSupportKind_LessThan = 5,
} EnzymeSupportKind;

MLIR_CAPI_EXPORTED MlirAttribute enzymeSupportAttrGet(
    MlirContext ctx, EnzymeSupportKind kind, bool hasLowerBound,
    double lowerBound, bool hasUpperBound, double upperBound);

MLIR_CAPI_EXPORTED MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx,
                                                        double trajectoryLength,
                                                        bool adaptStepSize,
                                                        bool adaptMassMatrix);

MLIR_CAPI_EXPORTED MlirAttribute enzymeNUTSConfigAttrGet(
    MlirContext ctx, int64_t maxTreeDepth, bool hasMaxDeltaEnergy,
    double maxDeltaEnergy, bool adaptStepSize, bool adaptMassMatrix);

MLIR_CAPI_EXPORTED MlirAttribute enzymeSymbolAttrGet(MlirContext ctx,
                                                     uint64_t ptr);

//===----------------------------------------------------------------------===//
// Custom forward-mode derivatives of calls
//===----------------------------------------------------------------------===//

/// Handle to the derivative that is being built, passed to a forward call
/// handler. It is valid only while the handler runs.
typedef struct EnzymeMLIRGradientUtils *EnzymeMLIRGradientUtilsRef;

/// Builds the forward-mode derivative of `call`, an op of the function being
/// differentiated, instead of differentiating its callee. The handler sets the
/// shadow of every active result of `call` (enzymeGradientUtilsSetDiffe). It
/// either keeps the call's clone in the derivative to compute the primal
/// results or replaces them (enzymeGradientUtilsReplaceOrigOpWith). New ops are
/// inserted before that clone with enzymeGradientUtilsInsertOperation.
typedef MlirLogicalResult (*EnzymeMLIRForwardCallHandler)(
    MlirOperation call, EnzymeMLIRGradientUtilsRef gutils, void *userData);

/// Registers `handler` under `key`, replacing any handler registered under it.
/// A direct call of a flat symbol uses the handler for its key: the
/// `enzyme_math` string attribute on the call, or else the one on the callee.
/// The callee's name is not a key. The registry is process-wide and not
/// synchronized: register and unregister handlers only while no pass that looks
/// them up is running, e.g. at startup. `userData` is passed to the handler.
MLIR_CAPI_EXPORTED void enzymeRegisterForwardCallHandler(
    MlirStringRef key, EnzymeMLIRForwardCallHandler handler, void *userData);

MLIR_CAPI_EXPORTED void enzymeUnregisterForwardCallHandler(MlirStringRef key);

/// The number of tangents computed at once (the batch width).
MLIR_CAPI_EXPORTED int64_t
enzymeGradientUtilsGetWidth(EnzymeMLIRGradientUtilsRef gutils);

/// Whether `value`, a value of the original function, is inactive (has no
/// derivative).
MLIR_CAPI_EXPORTED bool
enzymeGradientUtilsIsConstantValue(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirValue value);

/// The value in the derivative that corresponds to `value`, a value of the
/// original function.
MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsGetNewFromOriginal(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

/// The shadow (tangent) of `value`, a value of the original function. It is
/// zero if `value` is inactive.
MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsInvertPointer(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

/// The type of the shadow of a value of type `type`.
MLIR_CAPI_EXPORTED MlirType enzymeGradientUtilsGetShadowType(
    EnzymeMLIRGradientUtilsRef gutils, MlirType type);

/// Inserts the detached operation `op` into the derivative, after the ops that
/// the handler has already inserted. The derivative takes ownership of `op`.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsInsertOperation(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirOperation op);

/// Sets `shadow` as the shadow of the active result `result` of the call.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsSetDiffe(EnzymeMLIRGradientUtilsRef gutils, MlirValue result,
                            MlirValue shadow);

/// Replaces the call's results in the derivative with `values` (one per result)
/// and erases the call's clone. Operations inserted afterwards go where the
/// clone was.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsReplaceOrigOpWith(EnzymeMLIRGradientUtilsRef gutils,
                                     intptr_t nValues, MlirValue const *values);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
