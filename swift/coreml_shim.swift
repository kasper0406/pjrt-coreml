// Compile using: swiftc -emit-library -o swift/libswift_coreml.dylib -cxx-interoperability-mode=default swift/coreml_shim.swift -emit-clang-header-path swift/InternalBridge.hpp -Xcc -std=c++20

import CoreML

public enum DataType {
    case F16
    case F32
    case F64
    case I32

    func getWidth() -> Int {
        switch self {
            case .F16:
                return 2
            case .F32:
                return 4
            case .F64:
                return 8
            case .I32:
                return 4
        }
    }

    func getMLDataType() -> MLMultiArrayDataType {
        switch self {
            case .F16:
                return .float16
            case .F32:
                return .float32
            case .F64:
                return .float64
            case .I32:
                return .int32
        }
    }
};

public class Buffer {
    let array: MLMultiArray
    let dataPtr: UnsafeMutableRawPointer

    public convenience init(zeros dataType: DataType, _ shape: [UInt64], _ strides: [UInt64]) {
        self.init(internal: dataType, shape, strides, dataToCopy: nil)
    }
    
    public convenience init(withData dataPtr: UnsafeRawPointer, dataType: DataType, _ shape: [UInt64], _ strides: [UInt64]) {
        self.init(internal: dataType, shape, strides, dataToCopy: dataPtr)
    }

    private init(internal dataType: DataType, _ shape: [UInt64], _ strides: [UInt64], dataToCopy: Optional<UnsafeRawPointer>) {
        let byteCount = shape.reduce(1, { $0 * Int($1) }) * dataType.getWidth()
        print("mltensor_create: { shape: \(shape), strides: \(strides), bytesCount = \(byteCount) }")

        if shape.count != strides.count {
            fatalError("Attempted to create a buffer with undefined rank")
        }

        // Allocate the data to put in the MLTensor
        self.dataPtr = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: dataType.getWidth());

        if let sourcePtr = dataToCopy {
            // Copy the data from `dataToCopy`
            self.dataPtr.copyMemory(from: sourcePtr, byteCount: byteCount)
        } else {
            // Zero out the memory
            self.dataPtr.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)
        }

        // Construct an array
        do {
            self.array = try MLMultiArray(
                dataPointer: self.dataPtr,
                shape: shape as [NSNumber],
                dataType: dataType.getMLDataType(),
                strides: strides as [NSNumber]
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

    public func getRank() -> Int {
        return getShape().count
    }

    public func getShape() -> [Int] {
        array.shape.map { $0.intValue }
    }

    public func getStrides() -> [Int] {
        array.strides.map { $0.intValue }
    }

    public func getRawDataPointer() -> UnsafeMutableRawPointer {
        return self.dataPtr;
    }
}
