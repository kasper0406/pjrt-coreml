#pragma once

// #include "cxx.h"

#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

#include "InternalBridge.hpp"

namespace coreml {
    enum ElementType {
        F16, F32, F64, I32
    };

    class WrappedBuffer {
        swift_coreml::Buffer data;

    public:
        WrappedBuffer(ElementType dataType, size_t rank, const size_t* shape, const size_t* strides)
            : data([&]() -> swift_coreml::Buffer {
                auto swiftShape = swift::Array<uint64_t>::init();
                auto swiftStrides = swift::Array<uint64_t>::init();
                for (size_t i = 0; i < rank; i++) {
                    swiftShape.append(shape[i]);
                    swiftStrides.append(strides[i]);
                }

                return swift_coreml::Buffer::init(
                    toSwiftDataType(dataType),
                    swiftShape,
                    swiftStrides
                );
            }())
        {}
        
        WrappedBuffer(const void* dataPtr, ElementType dataType, size_t rank, const size_t* shape, const size_t* strides)
            : data([&]() -> swift_coreml::Buffer {
                auto swiftShape = swift::Array<uint64_t>::init();
                auto swiftStrides = swift::Array<uint64_t>::init();
                for (size_t i = 0; i < rank; i++) {
                    swiftShape.append(shape[i]);
                    swiftStrides.append(strides[i]);
                }

                return swift_coreml::Buffer::init(
                    dataPtr,
                    toSwiftDataType(dataType),
                    swiftShape,
                    swiftStrides
                );
            }())
        {}

        WrappedBuffer(WrappedBuffer&&) noexcept = default;
        WrappedBuffer& operator=(WrappedBuffer&&) noexcept = default;

        inline size_t getRank() {
            auto rank = data.getRank();
            return rank;
        }

        inline void getShape(size_t* output_shape) {
            auto shape = data.getShape();
            for (int i = 0; i < shape.getCount(); i++) {
                *(output_shape + i) = shape[i];
            }
        }

        inline void getStrides(size_t* output_strides) {
            auto strides = data.getStrides();
            for (int i = 0; i < strides.getCount(); i++) {
                *(output_strides + i) = strides[i];
            }
        }

        inline void* getRawDataPointer() {
            return data.getRawDataPointer();
        }

    private:
        inline swift_coreml::DataType toSwiftDataType(ElementType dt) {
            switch (dt) {
                case ElementType::F16: return swift_coreml::DataType::F16();
                case ElementType::F32: return swift_coreml::DataType::F32();
                case ElementType::F64: return swift_coreml::DataType::F64();
                case ElementType::I32: return swift_coreml::DataType::I32();
            }
        }
    };
}
