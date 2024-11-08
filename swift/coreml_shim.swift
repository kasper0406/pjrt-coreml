// Compile using: swiftc -emit-library -o swift/libswift_coreml.dylib -cxx-interoperability-mode=default swift/coreml_shim.swift -emit-clang-header-path swift/Bridge.hpp -Xcc -std=c++14

import CoreML
import Cxx
import CxxStdlib

public struct TensorShape {
    let shape0: Int
    let shape1: Int
    let shape2: Int
    let shape3: Int
    let shape4: Int
}

public class TensorDescription {
    // TODO(knielsen): Make this private!
    let mltensor_ptr: UnsafeMutableRawPointer
    let data_ptr: UnsafeMutableRawPointer

    // public let shape: (Int, Int, Int, Int, Int)

    init(_ mltensor_ptr: UnsafeMutableRawPointer, _ data_ptr: UnsafeMutableRawPointer) {
        self.mltensor_ptr = mltensor_ptr;

        self.data_ptr = data_ptr;
    }

    public func getShape(_ tensor: TensorDescription) -> TensorShape {
        let tensor = tensor.mltensor_ptr.load(as: MLMultiArray.self)
        let shape = tensor.shape.map { $0.intValue }
        return TensorShape(
            shape0: shape[0],
            shape1: shape[1],
            shape2: shape[2],
            shape3: shape[3],
            shape4: shape[4]
        )
    }
}

public func mltensor_create() -> TensorDescription {
    print("Inside swift mltensor_create!")

    // Allocate the data to put in the MLTensor
    let dataShape = [3]
    let dataPtr = UnsafeMutablePointer<Float32>.allocate(capacity: dataShape[0])
    dataPtr[0] = 1.5
    dataPtr[1] = 3.0
    dataPtr[2] = 5.0
    
    // Construct an array
    let mltensorPtr = UnsafeMutablePointer<MLMultiArray>.allocate(capacity: 1)
    do {
        mltensorPtr.initialize(to: try MLMultiArray(
            dataPointer: dataPtr,
            shape: dataShape as [NSNumber],
            dataType: .float32,
            strides: [1]
        ))

        return TensorDescription(mltensorPtr, dataPtr)
    } catch let error {
        print("Failed to allocate CoreML buffer!")
        mltensorPtr.deallocate();
        dataPtr.deallocate()

        fatalError("TODO: Handle this case!")
    }
}

// public func mltensor_shape(_ tensor: TensorDescription) -> [Int] {
//     let tensor = tensor.mltensor_ptr.load(as: MLMultiArray.self)
//     let shape = tensor.shape.map { $0.intValue }
//     return shape
// }

public func mltensor_destroy(_ tensor: TensorDescription) {
    print("Inside swift mltensor_destroy!")
    let tensor_ptr = tensor.mltensor_ptr.bindMemory(to: MLMultiArray.self, capacity: 1)
    tensor_ptr.deinitialize(count: 1)
    tensor_ptr.deallocate()

    tensor.data_ptr.deallocate()
}
