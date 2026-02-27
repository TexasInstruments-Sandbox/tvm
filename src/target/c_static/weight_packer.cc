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
 * \file weight_packer.cc
 * \brief Export VM constants/weights to an external binary file and generate
 *        C++ loader code for the c_static backend.
 *
 * This is a standalone utility that reads the public `constants` vector from
 * VMExecutable and serializes it using the same binary format as
 * SaveConstantSection, but to an external file rather than the embedded
 * bytecode stream.  Keeping this in c_static avoids modifying the core
 * executable.h/cc files.
 */
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/target/target.h>

#include "../../runtime/file_utils.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

using runtime::vm::VMExecutable;

/*!
 * \brief Serialize the constants vector to a dmlc::Stream.
 *
 * Mirrors VMExecutable::SaveConstantSection (which is private) but operates
 * on an externally-supplied constants vector.
 */
static void SerializeConstants(dmlc::Stream* strm,
                               const std::vector<ffi::Any>& constants) {
  strm->Write(static_cast<uint64_t>(constants.size()));
  for (const auto& it : constants) {
    if (auto opt_nd = it.as<runtime::Tensor>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFITensor);
      runtime::SaveDLTensor(strm, opt_nd.value().operator->());
    } else if (auto opt_shape = it.as<ffi::Shape>()) {
      ffi::Shape shape = opt_shape.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIShape);
      strm->Write(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        strm->Write(shape.at(i));
      }
    } else if (auto opt_str = it.as<ffi::String>()) {
      ffi::String str = opt_str.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIStr);
      strm->Write(str.size());
      for (size_t i = 0; i < str.size(); ++i) {
        strm->Write(str.at(i));
      }
    } else if (auto opt_int = it.as<int64_t>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIInt);
      strm->Write(opt_int.value());
    } else if (auto opt_float = it.as<double>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIFloat);
      strm->Write(opt_float.value());
    } else if (auto opt_dtype = it.as<DLDataType>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIDataType);
      strm->Write(opt_dtype.value());
    } else {
      LOG(FATAL) << "Unsupported constant pool type " << it.GetTypeKey();
    }
  }
}

/*!
 * \brief Serialize constants with alignment padding for DSP targets.
 *
 * For targets like TI C66x/C7x DSP that require aligned memory access,
 * this adds padding before Tensor entries to ensure data alignment.
 */
static void SerializeConstantsAligned(dmlc::Stream* strm,
                                      const std::vector<ffi::Any>& constants,
                                      std::string* data_buf,
                                      int alignment) {
  strm->Write(static_cast<uint64_t>(constants.size()));

  for (const auto& it : constants) {
    // For Tensor entries, add padding to ensure data alignment
    if (it.as<runtime::Tensor>()) {
      size_t current_pos = data_buf->size();
      size_t padding_needed = (alignment - (current_pos % alignment)) % alignment;
      for (size_t i = 0; i < padding_needed; i++) {
        uint8_t zero = 0;
        strm->Write(&zero, 1);
      }
    }

    // Write the constant (same logic as SerializeConstants)
    if (auto opt_nd = it.as<runtime::Tensor>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFITensor);
      runtime::SaveDLTensor(strm, opt_nd.value().operator->());
    } else if (auto opt_shape = it.as<ffi::Shape>()) {
      ffi::Shape shape = opt_shape.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIShape);
      strm->Write(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        strm->Write(shape.at(i));
      }
    } else if (auto opt_str = it.as<ffi::String>()) {
      ffi::String str = opt_str.value();
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIStr);
      strm->Write(str.size());
      for (size_t i = 0; i < str.size(); ++i) {
        strm->Write(str.at(i));
      }
    } else if (auto opt_int = it.as<int64_t>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIInt);
      strm->Write(opt_int.value());
    } else if (auto opt_float = it.as<double>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIFloat);
      strm->Write(opt_float.value());
    } else if (auto opt_dtype = it.as<DLDataType>()) {
      strm->Write<int32_t>(ffi::TypeIndex::kTVMFFIDataType);
      strm->Write(opt_dtype.value());
    } else {
      LOG(FATAL) << "Unsupported constant pool type " << it.GetTypeKey();
    }
  }
}

/*!
 * \brief Save constants to an external binary file.
 */
static void SaveConstantSectionToFile(const std::vector<ffi::Any>& constants,
                                      const std::string& filename) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  SerializeConstants(&writer, constants);

  try {
    runtime::SaveBinaryToFile(filename, data);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to save constants to " << filename << ": " << e.what();
  }
}

/*!
 * \brief Save constants to an external binary file with alignment.
 */
static void SaveConstantSectionToFileAligned(const std::vector<ffi::Any>& constants,
                                             const std::string& filename,
                                             int alignment) {
  std::string data;
  dmlc::MemoryStringStream writer(&data);
  SerializeConstantsAligned(&writer, constants, &data, alignment);

  try {
    runtime::SaveBinaryToFile(filename, data);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to save constants to " << filename << ": " << e.what();
  }
}

/*!
 * \brief Generate C++ code that loads constants from an external binary file.
 */
static void GenerateConstantLoaderCode(std::ostream& os,
                                       const std::string& filename) {
  std::string basename = filename.substr(filename.find_last_of("/\\") + 1);
  // NOTE: The generated code below is compiled as part of the c_static binary,
  // NOT as part of TVM itself.  For DSP targets, the DSP runtime provides its
  // own TVMGetConstants() in constants_loader.cpp, so this generated code is
  // only used for generic host-emulation builds.
  os << R"(
// Auto-generated constant loader for external binary file.
// This file is compiled into the c_static deployment binary.
// For DSP targets, the DSP runtime provides its own TVMGetConstants().
#include <dmlc/memory_io.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/tensor.h>
#include <fstream>
#include <string>
#include <vector>

std::vector<tvm::ffi::Any> TVMGetConstants() {
  static std::vector<tvm::ffi::Any> cached_constants;
  static bool constants_loaded = false;

  if (!constants_loaded) {
    std::string filename = ")" << basename << R"(";

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      LOG(FATAL) << "Failed to open constants file: " << filename;
      return cached_constants;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string data;
    data.resize(file_size);
    file.read(&data[0], file_size);
    file.close();

    // Deserialize using the same binary format as VM SaveConstantSection
    dmlc::MemoryStringStream stream(&data);
    uint64_t sz;
    stream.Read(&sz, sizeof(sz));
    tvm::runtime::Tensor ndarray;
    DLDataType dtype;
    for (uint64_t i = 0; i < sz; i++) {
      int32_t constant_type;
      stream.Read(&constant_type, sizeof(constant_type));
      if (constant_type == tvm::ffi::TypeIndex::kTVMFFITensor) {
        ndarray.Load(&stream);
        tvm::ffi::Any cell;
        cell = ndarray;
        cached_constants.push_back(cell);
      } else if (constant_type == tvm::ffi::TypeIndex::kTVMFFIShape) {
        uint64_t len;
        stream.Read(&len, sizeof(len));
        std::vector<tvm::ffi::Shape::index_type> sd(len);
        for (uint64_t j = 0; j < len; ++j) stream.Read(&(sd[j]), sizeof(sd[j]));
        tvm::ffi::Any cell;
        cell = tvm::ffi::Shape(sd);
        cached_constants.push_back(cell);
      } else if (constant_type == tvm::ffi::TypeIndex::kTVMFFIStr) {
        uint64_t len;
        stream.Read(&len, sizeof(len));
        std::string str_data(len, '\0');
        for (uint64_t j = 0; j < len; ++j) stream.Read(&str_data[j], 1);
        tvm::ffi::Any cell;
        cell = tvm::ffi::String(str_data);
        cached_constants.push_back(cell);
      } else if (constant_type == tvm::ffi::TypeIndex::kTVMFFIInt) {
        int64_t value;
        stream.Read(&value, sizeof(value));
        tvm::ffi::Any cell;
        cell = value;
        cached_constants.push_back(cell);
      } else if (constant_type == tvm::ffi::TypeIndex::kTVMFFIFloat) {
        double value;
        stream.Read(&value, sizeof(value));
        tvm::ffi::Any cell;
        cell = value;
        cached_constants.push_back(cell);
      } else if (constant_type == tvm::ffi::TypeIndex::kTVMFFIDataType) {
        stream.Read(&dtype, sizeof(dtype));
        tvm::ffi::Any cell;
        cell = dtype;
        cached_constants.push_back(cell);
      }
    }
    constants_loaded = true;
  }

  return cached_constants;
}
)";
}

/*!
 * \brief Export constants/weights from VMExecutable to an external binary
 *   file and generate C++ loader code.
 *
 * \param mod VMExecutable module containing constants
 * \param const_filename Path for the binary weights file
 * \param target Optional target for alignment control
 * \return Generated C++ code string
 */
std::string PackWeightsToBinary(const ffi::Module& mod,
                                const std::string& const_filename,
                                ffi::Optional<Target> target) {
  auto vm = mod.as<VMExecutable>();
  ICHECK(vm != nullptr) << "Module must be a VMExecutable for weight packing";

  std::ostringstream os;

  // Determine alignment based on target mcpu
  int alignment = 1;  // Default: no alignment padding
  if (target.has_value()) {
    auto mcpu = target.value()->GetAttr<ffi::String>("mcpu");
    if (mcpu.has_value()) {
      std::string mcpu_str = mcpu.value();
      // TI C66x/C7x DSP requires 4-byte alignment for data access
      if (mcpu_str.find("c66") == 0 || mcpu_str.find("c7") == 0) {
        alignment = 4;
      }
    }
  }

  // Save constants to external binary file
  if (alignment > 1) {
    SaveConstantSectionToFileAligned(vm->constants, const_filename, alignment);
  } else {
    SaveConstantSectionToFile(vm->constants, const_filename);
  }

  // Generate C++ loader code
  GenerateConstantLoaderCode(os, const_filename);

  return os.str();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.ModulePackWeightsToBinary",
      [](const ffi::Module& mod, const std::string& const_filename,
         ffi::Optional<Target> target) {
        return PackWeightsToBinary(mod, const_filename, target);
      });
}

}  // namespace codegen
}  // namespace tvm
