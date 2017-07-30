// Copyright 2012 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if V8_TARGET_ARCH_IA32

#include "src/interface-descriptors.h"

namespace v8 {
namespace internal {

const Register CallInterfaceDescriptor::ContextRegister() { return esi; }

void CallInterfaceDescriptor::DefaultInitializePlatformSpecific(
    CallInterfaceDescriptorData* data, int register_parameter_count) {
  const Register default_stub_registers[] = {eax, ebx, ecx, edx, edi};
  CHECK_LE(static_cast<size_t>(register_parameter_count),
           arraysize(default_stub_registers));
  data->InitializePlatformSpecific(register_parameter_count,
                                   default_stub_registers);
}

const Register FastNewFunctionContextDescriptor::FunctionRegister() {
  return edi;
}
const Register FastNewFunctionContextDescriptor::SlotsRegister() { return eax; }

const Register LoadDescriptor::ReceiverRegister() { return edx; }
const Register LoadDescriptor::NameRegister() { return ecx; }
const Register LoadDescriptor::SlotRegister() { return eax; }

const Register LoadWithVectorDescriptor::VectorRegister() { return ebx; }

const Register LoadICProtoArrayDescriptor::HandlerRegister() { return edi; }

const Register StoreDescriptor::ReceiverRegister() { return edx; }
const Register StoreDescriptor::NameRegister() { return ecx; }
const Register StoreDescriptor::ValueRegister() { return eax; }
const Register StoreDescriptor::SlotRegister() { return edi; }

const Register StoreWithVectorDescriptor::VectorRegister() { return ebx; }

const Register StoreTransitionDescriptor::SlotRegister() { return no_reg; }
const Register StoreTransitionDescriptor::VectorRegister() { return ebx; }
const Register StoreTransitionDescriptor::MapRegister() { return edi; }

const Register StringCompareDescriptor::LeftRegister() { return edx; }
const Register StringCompareDescriptor::RightRegister() { return eax; }

const Register ApiGetterDescriptor::HolderRegister() { return ecx; }
const Register ApiGetterDescriptor::CallbackRegister() { return eax; }

const Register MathPowTaggedDescriptor::exponent() { return eax; }

const Register MathPowIntegerDescriptor::exponent() {
  return MathPowTaggedDescriptor::exponent();
}

const Register RegExpExecDescriptor::StringRegister() { return eax; }
const Register RegExpExecDescriptor::LastIndexRegister() { return ecx; }
const Register RegExpExecDescriptor::StringStartRegister() { return edx; }
const Register RegExpExecDescriptor::StringEndRegister() { return ebx; }
const Register RegExpExecDescriptor::CodeRegister() { return edi; }

const Register GrowArrayElementsDescriptor::ObjectRegister() { return eax; }
const Register GrowArrayElementsDescriptor::KeyRegister() { return ebx; }


void FastNewClosureDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // SharedFunctionInfo, vector, slot index.
  Register registers[] = {ebx, ecx, edx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

// static
const Register TypeConversionDescriptor::ArgumentRegister() { return eax; }

void TypeofDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void FastCloneRegExpDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edi, eax, ecx, edx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void FastCloneShallowArrayDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {eax, ebx, ecx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void FastCloneShallowObjectDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {eax, ebx, ecx, edx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void CreateAllocationSiteDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {ebx, edx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void CreateWeakCellDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {ebx, edx, edi};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void CallFunctionDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edi};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void CallICTrampolineDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edi, eax, edx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void CallICDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edi, eax, edx, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void CallConstructDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // eax : number of arguments
  // ebx : feedback vector
  // ecx : new target (for IsSuperConstructorCall)
  // edx : slot in feedback vector (Smi, for RecordCallTarget)
  // edi : constructor function
  // TODO(turbofan): So far we don't gather type feedback and hence skip the
  // slot parameter, but ArrayConstructStub needs the vector to be undefined.
  Register registers[] = {eax, edi, ecx, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void CallTrampolineDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // eax : number of arguments
  // edi : the target to call
  Register registers[] = {edi, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void CallForwardVarargsDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // ecx : start index (to support rest parameters)
  // edi : the target to call
  Register registers[] = {edi, ecx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void ConstructStubDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // eax : number of arguments
  // edx : the new target
  // edi : the target to call
  // ebx : allocation site or undefined
  Register registers[] = {edi, edx, eax, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void ConstructTrampolineDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // eax : number of arguments
  // edx : the new target
  // edi : the target to call
  Register registers[] = {edi, edx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}


void TransitionElementsKindDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {eax, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void AllocateHeapNumberDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // register state
  data->InitializePlatformSpecific(0, nullptr, nullptr);
}

void ArrayConstructorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // kTarget, kNewTarget, kActualArgumentsCount, kAllocationSite
  Register registers[] = {edi, edx, eax, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void ArrayNoArgumentConstructorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // register state
  // eax -- number of arguments
  // edi -- function
  // ebx -- allocation site with elements kind
  Register registers[] = {edi, ebx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void ArraySingleArgumentConstructorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // register state
  // eax -- number of arguments
  // edi -- function
  // ebx -- allocation site with elements kind
  Register registers[] = {edi, ebx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void ArrayNArgumentsConstructorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // register state
  // eax -- number of arguments
  // edi -- function
  // ebx -- allocation site with elements kind
  Register registers[] = {edi, ebx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void VarArgFunctionDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // stack param count needs (arg count)
  Register registers[] = {eax};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void CompareDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void BinaryOpDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}


void BinaryOpWithAllocationSiteDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {ecx, edx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void BinaryOpWithVectorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  // register state
  // edx -- lhs
  // eax -- rhs
  // edi -- slot id
  // ebx -- vector
  Register registers[] = {edx, eax, edi, ebx};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void CountOpDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {eax};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void StringAddDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {edx, eax};
  data->InitializePlatformSpecific(arraysize(registers), registers, NULL);
}

void ArgumentAdaptorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      edi,  // JSFunction
      edx,  // the new target
      eax,  // actual number of arguments
      ebx,  // expected number of arguments
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void ApiCallbackDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      edi,  // callee
      ebx,  // call_data
      ecx,  // holder
      edx,  // api_function_address
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void InterpreterDispatchDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      kInterpreterAccumulatorRegister, kInterpreterBytecodeOffsetRegister,
      kInterpreterBytecodeArrayRegister, kInterpreterDispatchTableRegister};
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void InterpreterPushArgsThenCallDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      eax,  // argument count (not including receiver)
      ebx,  // address of first argument
      edi   // the target callable to be call
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void InterpreterPushArgsThenConstructDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      eax,  // argument count (not including receiver)
      edx,  // new target
      edi,  // constructor
      ebx,  // allocation site feedback
      ecx,  // address of first argument
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void InterpreterPushArgsThenConstructArrayDescriptor::
    InitializePlatformSpecific(CallInterfaceDescriptorData* data) {
  Register registers[] = {
      eax,  // argument count (not including receiver)
      edx,  // target to the call. It is checked to be Array function.
      ebx,  // allocation site feedback
      ecx,  // address of first argument
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void InterpreterCEntryDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      eax,  // argument count (argc)
      ecx,  // address of first argument (argv)
      ebx   // the runtime function to call
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void ResumeGeneratorDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      eax,  // the value to pass to the generator
      ebx,  // the JSGeneratorObject to resume
      edx,  // the resume mode (tagged)
      ecx   // SuspendFlags (tagged)
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

void FrameDropperTrampolineDescriptor::InitializePlatformSpecific(
    CallInterfaceDescriptorData* data) {
  Register registers[] = {
      ebx,  // loaded new FP
  };
  data->InitializePlatformSpecific(arraysize(registers), registers);
}

}  // namespace internal
}  // namespace v8

#endif  // V8_TARGET_ARCH_IA32