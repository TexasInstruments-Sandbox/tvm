/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_c_static_wrapper.cc
 * \brief Implementation of wrapper function generator.
 */
#include "codegen_c_static_wrapper.h"

#include <cstring>

namespace tvm {
namespace codegen {

std::string WrapperGenerator::GetWrapperName(const std::string& tir_func_name, bool dsp_suffix) {
  std::string wrapper_name = tir_func_name;
  const size_t prefix_len = std::strlen(kVMTIRPrefix);
  if (wrapper_name.substr(0, prefix_len) == kVMTIRPrefix) {
    wrapper_name = wrapper_name.substr(prefix_len);
  }
  wrapper_name = std::string(kWrapperPrefix) + wrapper_name;
  if (dsp_suffix) {
    wrapper_name += "_dsp";
  }
  return wrapper_name;
}

void WrapperGenerator::EmitStandardWrappers(
    const std::unordered_map<std::string, FunctionInfo>& functions,
    const std::map<std::string, std::string>& vm_builtins,
    std::ostream& os) {

  os << "\n// Auto-generated wrapper functions\n";

  for (const auto& [func_name, func_info] : functions) {
    if (func_info.was_private) {
      continue;  // Skip private functions - they're implementation details
    }

    std::string wrapper_name = GetWrapperName(func_name, false);
    int reg_file_size = func_info.max_register_index + 1;

    // Primary interface with explicit parameters
    os << "// Primary interface (explicit parameters)\n";
    if (func_info.returns_tuple) {
      os << "Array<NDArray> " << wrapper_name << "(";
    } else {
      os << "NDArray " << wrapper_name << "(";
    }
    for (int i = 0; i < func_info.num_args; ++i) {
      if (i > 0) os << ", ";
      os << "NDArray& arg" << i;
    }
    os << ") {\n";

    // Generate function body
    os << "  std::vector<tvm::ffi::Any> constants = TVMGetConstants();\n\n";
    os << "  // The reg_file_size is CGFunctionInfo.max_register_index+1\n";
    os << "  const int reg_file_size = " << reg_file_size << ";\n";
    os << "  std::vector<tvm::ffi::Any> reg_file(reg_file_size);\n";

    // Initialize register file with input arguments
    for (int i = 0; i < func_info.num_args; ++i) {
      os << "  reg_file[" << i << "] = arg" << i << ";\n";
    }
    os << "\n";

    // Initialize VM with exception safety
    os << "  // Initialize a VM with allocators and pass as the context pointer to "
       << func_name << "\n";
    os << "  tvm::Device device{kDLCPU, 0};\n";
    os << "  auto vm = tvm::runtime::vm::VirtualMachine::Create();\n";
    os << "  if (!vm) {\n";
    os << "    throw std::runtime_error(\"Failed to create VirtualMachine\");\n";
    os << "  }\n";
    os << "  try {\n";
    os << "    vm->InitAllocators({device}, {AllocatorType::kPooled});\n";
    os << "  } catch (const std::exception& e) {\n";
    os << "    throw std::runtime_error(\"Failed to initialize VM allocators: \""
       << " + std::string(e.what()));\n";
    os << "  }\n\n";

    // Initialize VM builtins if we have any
    if (!vm_builtins.empty()) {
      os << "  // Initialize VM builtin functions\n";
      os << "  if (InitVMBuiltins() != 0) {\n";
      os << "    throw std::runtime_error(\"Failed to initialize VM builtin functions\");\n";
      os << "  }\n\n";
    }

    // Set up arguments
    os << "  // Set up the arguments for " << func_name
       << ", always has size of " << kTIRArgsSize << "\n";
    os << "  const int args_size = " << kTIRArgsSize << ";\n";
    os << "  std::vector<tvm::ffi::Any> args(args_size);\n";
    os << "  args[" << kTIRVMContextIndex << "] = static_cast<void*>(vm.get());\n";
    os << "  args[" << kTIRRegFileIndex << "] = static_cast<void*>(reg_file.data());\n";
    os << "  args[" << kTIRConstantsIndex << "] = static_cast<void*>(constants.data());\n";
    os << "  args[" << kTIRReservedIndex << "] = nullptr;\n";

    // Call the TIR function and check return value
    os << "  // Call the TIR function and check for errors\n";
    os << "  TVMFFIAny ret_val;\n";
    os << "  ret_val.type_index = kTVMFFINone;\n";
    os << "  ret_val.v_int64 = 0;\n";
    os << "  int ret_code = " << func_name
       << "(nullptr, args.data(), args_size, &ret_val);\n";
    os << "  if (ret_code != 0) {\n";
    os << "    // TIR function returned an error. The detailed error message is stored in\n";
    os << "    // thread-local storage by TVM_FFI_SAFE_CALL_END. Retrieve and re-throw it.\n";
    os << "    // This pattern matches TVM_FFI_CHECK_SAFE_CALL macro.\n";
    os << "    if (ret_code == -2) {\n";
    os << "      throw tvm::ffi::EnvErrorAlreadySet();\n";
    os << "    }\n";
    os << "    throw tvm::ffi::details::MoveFromSafeCallRaised();\n";
    os << "  }\n";

    // Extract result based on return type
    os << "  // The output is stored in the register file after the inputs\n";
    os << "  // Use .as<T>().value() to cast from tvm::ffi::Any to target type\n";
    if (func_info.returns_tuple) {
      os << "  Array<NDArray> out = reg_file[" << func_info.num_args
         << "].as<Array<NDArray>>().value();\n";
    } else {
      os << "  NDArray out = reg_file[" << func_info.num_args
         << "].as<NDArray>().value();\n";
    }

    os << "  return std::move(out);  // Transfer ownership, avoid reference count increment\n";
    os << "}\n\n";

    // Convenience overload for dynamic calling with Array
    os << "// Convenience overload for dynamic calling\n";
    if (func_info.returns_tuple) {
      os << "Array<NDArray> " << wrapper_name << "(const Array<NDArray>& args) {\n";
    } else {
      os << "NDArray " << wrapper_name << "(const Array<NDArray>& args) {\n";
    }

    os << "  if (args.size() != " << func_info.num_args << ") {\n";
    os << "    throw std::invalid_argument(\"Expected " << func_info.num_args
       << " args, got \" + std::to_string(args.size()));\n";
    os << "  }\n";

    // Create local copies (cheap due to reference counting)
    for (int i = 0; i < func_info.num_args; ++i) {
      os << "  NDArray arg" << i << " = args[" << i << "];\n";
    }

    os << "  return " << wrapper_name << "(";
    for (int i = 0; i < func_info.num_args; ++i) {
      if (i > 0) os << ", ";
      os << "arg" << i;
    }
    os << ");\n";

    os << "}\n\n";
  }
}

void WrapperGenerator::EmitDSPWrappers(
    const std::unordered_map<std::string, FunctionInfo>& functions,
    const std::map<std::string, std::string>& vm_builtins,
    std::ostream& os) {

  os << "\n// ============================================================\n";
  os << "// DSP Wrapper Functions (C66x/C7x compatible)\n";
  os << "// ============================================================\n\n";

  for (const auto& [func_name, func_info] : functions) {
    if (func_info.was_private) {
      continue;  // Skip private functions
    }

    std::string wrapper_name = GetWrapperName(func_name, true);
    int reg_file_size = func_info.max_register_index + 1;

    // Emit the wrapper function
    os << "/*!\n";
    os << " * \\brief DSP inference wrapper for " << func_name << "\n";
    os << " *\n";
    os << " * \\param inputs Array of " << func_info.num_args << " input TVMFFIAny values\n";
    os << " * \\param num_inputs Number of inputs (must be " << func_info.num_args << ")\n";
    os << " * \\param constants Constants array from TVMDSPConstantsGet()\n";
    os << " * \\param output Pointer to receive output TVMFFIAny\n";
    os << " * \\return 0 on success, negative error code on failure\n";
    os << " *\n";
    os << " * \\note Register file is statically allocated (size=" << reg_file_size << ")\n";
    os << " * \\note NOT thread-safe - single inference at a time\n";
    os << " */\n";
    os << "extern \"C\" TVM_DSP_EXPORT int " << wrapper_name << "(\n";
    os << "    TVMFFIAny* inputs, int num_inputs,\n";
    os << "    TVMFFIAny* constants, TVMFFIAny* output) {\n";

    // Validate input count
    os << "  // Validate input count\n";
    os << "  if (num_inputs != " << func_info.num_args << ") {\n";
    os << "    return -1;  // Invalid input count\n";
    os << "  }\n\n";

    // Reset L2 SRAM bump allocator (reclaims all L2 scratch from previous inference)
    os << "  tvm_l2_reset();\n\n";

    // Static register file
    os << "  // Static register file (avoids heap allocation)\n";
    os << "  static TVMFFIAny reg_file[" << reg_file_size << "];\n";
    os << "  memset(reg_file, 0, sizeof(reg_file));\n";
    os << "  TVMDSPRegFileInit(reg_file, " << reg_file_size << ");\n\n";

    // Initialize register file with inputs
    os << "  // Initialize register file with inputs\n";
    for (int i = 0; i < func_info.num_args; ++i) {
      os << "  reg_file[" << i << "] = inputs[" << i << "];\n";
    }
    os << "\n";

    // Initialize VM builtins if needed
    if (!vm_builtins.empty()) {
      os << "  // Register and initialize VM builtin functions\n";
      os << "  if (TVMDSPRegisterVMBuiltins() != 0) {\n";
      os << "    return -2;  // Failed to register builtins\n";
      os << "  }\n";
      os << "  if (InitVMBuiltins() != 0) {\n";
      os << "    return -3;  // Failed to initialize builtin pointers\n";
      os << "  }\n\n";
    }

    // Set up TIR arguments
    os << "  // Set up arguments for " << func_name << "\n";
    os << "  TVMFFIAny args[" << kTIRArgsSize << "];\n";
    os << "  memset(args, 0, sizeof(args));\n";
    os << "  args[" << kTIRVMContextIndex << "].type_index = kTVMFFIOpaquePtr;\n";
    os << "  args[" << kTIRVMContextIndex << "].v_ptr = nullptr;  // VM context (unused on DSP)\n";
    os << "  args[" << kTIRRegFileIndex << "].type_index = kTVMFFIOpaquePtr;\n";
    os << "  args[" << kTIRRegFileIndex << "].v_ptr = static_cast<void*>(reg_file);\n";
    os << "  args[" << kTIRConstantsIndex << "].type_index = kTVMFFIOpaquePtr;\n";
    os << "  args[" << kTIRConstantsIndex << "].v_ptr = static_cast<void*>(constants);\n";
    os << "  args[" << kTIRReservedIndex << "].type_index = kTVMFFIOpaquePtr;\n";
    os << "  args[" << kTIRReservedIndex << "].v_ptr = nullptr;  // Reserved\n\n";

    // Call the TIR function
    os << "  // Call the TIR function\n";
    os << "  TVMFFIAny ret_val;\n";
    os << "  ret_val.type_index = kTVMFFINone;\n";
    os << "  ret_val.v_int64 = 0;\n";
    os << "  int ret_code = " << func_name << "(nullptr, args, "
       << kTIRArgsSize << ", &ret_val);\n";
    os << "  if (ret_code != 0) {\n";
    os << "    return ret_code;  // TIR function error\n";
    os << "  }\n\n";

    // Copy output from register file
    os << "  // Copy output from register file\n";
    os << "  // Output is at reg_file[" << func_info.num_args << "] (after inputs)\n";
    os << "  if (output != nullptr) {\n";
    os << "    *output = reg_file[" << func_info.num_args << "];\n";
    os << "  }\n\n";

    os << "  return 0;  // Success\n";
    os << "}\n\n";
  }
}

}  // namespace codegen
}  // namespace tvm
