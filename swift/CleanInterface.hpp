#pragma once

#include <vector>
#include <memory>
#include <cstdint>

#include "InternalBridge.hpp"

namespace coreml {
    class WrappedBuffer {
        // std::unique_ptr<swift_coreml::Buffer> data;
        swift_coreml::Buffer data;

    public:
        WrappedBuffer() : data(swift_coreml::Buffer::init()) {
            // this->data = std::make_unique<swift_coreml::Buffer>(swift_coreml::Buffer::init());
        }

        WrappedBuffer(WrappedBuffer&&) noexcept = default;
        WrappedBuffer& operator=(WrappedBuffer&&) noexcept = default;

        // inline std::vector<int64_t> getShape();
        inline int64_t getShape();

        inline void* getRawDataPointer() {
            return data.getRawDataPointer();
        }
    };

    inline int64_t WrappedBuffer::getShape()  {
        // auto shape = std::make_unique<std::vector<int64_t>>();
        std::vector<int64_t> shape;
        for (int64_t i : data.getShape()) {
            shape.push_back(i);
        }
        return shape[0];
    }
}
