// Compile using: swiftc -emit-library -o swift/libswift_coreml.dylib -cxx-interoperability-mode=default swift/coreml_shim.swift -emit-clang-header-path swift/InternalBridge.hpp -Xcc -std=c++20

import CoreML

public class Buffer {
    let array: MLMultiArray
    let dataPtr: UnsafeMutableRawPointer

    public init() {
        print("Inside swift mltensor_create!")
        // Allocate the data to put in the MLTensor
        let dataShape = [3]
        let bufferPtr = UnsafeMutablePointer<Float32>.allocate(capacity: dataShape[0])
        bufferPtr[0] = 1.5
        bufferPtr[1] = 3.0
        bufferPtr[2] = 5.0
        self.dataPtr = UnsafeMutableRawPointer(bufferPtr)

        // Construct an array
        do {
            self.array = try MLMultiArray(
                dataPointer: self.dataPtr,
                shape: dataShape as [NSNumber],
                dataType: .float32,
                strides: [1]
            )
        } catch let error {
            print("Failed to allocate CoreML buffer!")
            self.dataPtr.deallocate()

            fatalError("TODO: Handle this case!")
        }
    }

    deinit {
        print("Inside swift mltensor_destroy!")
        self.dataPtr.deallocate()
    }

    public func getShape() -> [Int] {
        array.shape.map { $0.intValue }
    }

    public func getRawDataPointer() -> UnsafeMutableRawPointer {
        return self.dataPtr;
    }
}
