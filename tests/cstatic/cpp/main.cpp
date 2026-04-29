#include "cnpy.h"
#include <tvm/ffi/container/array.h>
#include <tvm/runtime/tensor.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Use two-level macro expansion to properly concatenate tokens
#define CONCAT_HELPER(a, b) a ## b
#define CONCAT(a, b) CONCAT_HELPER(a, b)

// Create function name: cg_ + MODEL_ENTRY_FUNCTION
#define CG_FUNC_NAME CONCAT(cg_, MODEL_ENTRY_FUNCTION)

// Error if not specified
#ifndef MODEL_NUM_INPUTS
#error "MODEL_NUM_INPUTS not specified"
#endif

#ifdef MODEL_RETURNS_TUPLE
extern tvm::ffi::Array<tvm::runtime::Tensor> CG_FUNC_NAME(const tvm::ffi::Array<tvm::runtime::Tensor>& args);
#else
extern tvm::runtime::Tensor CG_FUNC_NAME(const tvm::ffi::Array<tvm::runtime::Tensor>& args);
#endif

tvm::runtime::Tensor load_input_file(const std::string& filename, tvm::Device device) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    float* loaded_data = arr.data<float>();

    auto f32 = tvm::runtime::DataType::Float(32);
    std::vector<int64_t> numpy_shape(arr.shape.begin(), arr.shape.end());
    tvm::ffi::Shape tvm_shape(numpy_shape);

    tvm::runtime::Tensor input = tvm::runtime::Tensor::Empty(tvm_shape, f32, device);
    auto shape = input.Shape();
    int numel = 1;
    for (size_t i = 0; i < shape.size(); ++i) numel *= shape[i];
    for (int i = 0; i < numel; ++i)
        static_cast<float*>(input->data)[i] = loaded_data[i];

    return input;
}

// Helper function to extract shape from TVM NDArray
std::vector<size_t> get_shape_from_tensor(const tvm::runtime::Tensor& arr) {
    std::vector<size_t> shape;
    auto sv = arr.Shape();
    for (size_t i = 0; i < sv.size(); ++i) {
        shape.push_back(static_cast<size_t>(sv[i]));
    }
    return shape;
}

// Template helper to save NDArray with specific dtype
template<typename T>
void save_tensor_to_npz_typed(
    const std::string& zipname,
    const std::string& arrayname,
    const tvm::runtime::Tensor& arr,
    const std::string& mode
) {
    std::vector<size_t> shape = get_shape_from_tensor(arr);
    const T* data = static_cast<const T*>(arr->data);
    cnpy::npz_save(zipname, arrayname, data, shape, mode);
}

// Dispatcher based on TVM dtype
void save_output_to_npz(
    const std::string& zipname,
    const std::string& arrayname,
    const tvm::runtime::Tensor& arr,
    const std::string& mode
) {
    auto dtype = arr.DataType();
    
    if (dtype == tvm::runtime::DataType::Float(32)) {
        save_tensor_to_npz_typed<float>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::Float(16)) {
        // Store float16 as uint16, NumPy will interpret correctly
        save_tensor_to_npz_typed<uint16_t>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::Int(32)) {
        save_tensor_to_npz_typed<int32_t>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::Int(64)) {
        save_tensor_to_npz_typed<int64_t>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::Int(8)) {
        save_tensor_to_npz_typed<int8_t>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::UInt(8)) {
        save_tensor_to_npz_typed<uint8_t>(zipname, arrayname, arr, mode);
    } else if (dtype == tvm::runtime::DataType::Bool()) {
        save_tensor_to_npz_typed<bool>(zipname, arrayname, arr, mode);
    } else {
        std::ostringstream err;
        err << "Unsupported output dtype: " << dtype;
        throw std::runtime_error(err.str());
    }
}

int main(int argc, char** argv) {
    tvm::Device device{kDLCPU, 0};

    // Load multiple inputs based on compile-time constant
    std::vector<tvm::runtime::Tensor> inputs;
    inputs.reserve(MODEL_NUM_INPUTS);

    for (int i = 0; i < MODEL_NUM_INPUTS; ++i) {
        std::ostringstream filename;
        filename << "input_" << i << ".npy";
        inputs.push_back(load_input_file(filename.str(), device));
    }

    // Create Array from vector
    tvm::ffi::Array<tvm::runtime::Tensor> args(inputs.begin(), inputs.end());

    try {
        // Collect ALL outputs into Array
        #ifdef MODEL_RETURNS_TUPLE
        tvm::ffi::Array<tvm::runtime::Tensor> outputs = CG_FUNC_NAME(args);
        #else
        tvm::ffi::Array<tvm::runtime::Tensor> outputs;
        outputs.push_back(CG_FUNC_NAME(args));
        #endif

        // Save all outputs to NPZ file
        const std::string output_file = "outputs.npz";
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::ostringstream name;
            name << "output_" << i;
            
            // First write creates file ("w"), subsequent appends ("a")
            std::string mode = (i == 0) ? "w" : "a";
            save_output_to_npz(output_file, name.str(), outputs[i], mode);
        }

        return 0;

    } catch (const std::exception& e) {
        // Print error to stderr so it can be captured by the test framework
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        // Catch any other exceptions
        std::cerr << "Error: Unknown exception occurred" << std::endl;
        return 2;
    }
}
