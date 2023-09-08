#ifndef FREYR_METAL_BUFFER_IMPL_H
#define FREYR_METAL_BUFFER_IMPL_H

#include "../_buffer.h"
#include "../device_traits.h"
#include "core.h"

namespace freyr::device {

    class metal_buffer : public base_buffer<metal_buffer, detail::backend_types::METAL> {
    public:
        using base_buffer<metal_buffer, detail::backend_types::METAL>::base_buffer;

        // Existing constructor
        metal_buffer(MTL::Device         *device,
                     size_t               buffer_size,
                     MTL::ResourceOptions storage_mode = MTL::ResourceStorageModeShared,
                     unsigned int         usage_flags  = 0)
            : base_buffer<metal_buffer, detail::backend_types::METAL>(buffer_size, usage_flags),
              m_device(device) {
            m_buffer = m_device->newBuffer(buffer_size, storage_mode);
        }

        // Deleted copy constructor and assignment operator
        metal_buffer(const metal_buffer &)            = delete;
        metal_buffer &operator=(const metal_buffer &) = delete;

        // Move constructor
        metal_buffer(metal_buffer &&other) noexcept
            : base_buffer<metal_buffer, detail::backend_types::METAL>(other.size(),
                                                                      other.usage_flags()),
              m_device(other.m_device), m_buffer(other.m_buffer) {
            other.m_device = nullptr;
            other.m_buffer = nullptr;
        }

        // Move assignment operator
        metal_buffer &operator=(metal_buffer &&other) noexcept {
            if (this != &other) {
                m_device       = other.m_device;
                m_buffer       = other.m_buffer;
                other.m_device = nullptr;
                other.m_buffer = nullptr;
            }
            return *this;
        }

        ~metal_buffer() = default;

        void *contents_impl() { return m_buffer->contents(); }

        template<typename T>
        void set_data_impl(const std::vector<T> &data) {
            std::memcpy(m_buffer->contents(), data.data(), data.size() * sizeof(T));
        }

        template<typename T>
        std::vector<T> get_data_impl(size_t num_elements) {
            auto *buffer_contents = static_cast<T *>(m_buffer->contents());
            return std::vector<T>(buffer_contents, buffer_contents + num_elements);
        }

        [[nodiscard]] MTL::Buffer *native_buffer() const { return m_buffer; }

        // Implement copy_from_host for metal_buffer
        template<typename T>
        void copy_from_host_impl(const std::vector<T> &data) {
            std::memcpy(m_buffer->contents(), data.data(), data.size() * sizeof(T));
        }

        // Implement copy_to_host for metal_buffer
        template<typename T>
        std::vector<T> copy_to_host_impl(size_t num_elements) {
            auto *buffer_contents = static_cast<float *>(m_buffer->contents());
            return std::vector<T>(buffer_contents, buffer_contents + num_elements);
        }

    private:
        MTL::Device *m_device;
        MTL::Buffer *m_buffer;
    };


}// namespace freyr::device


#endif// FREYR_METAL_BUFFER_IMPL_H
