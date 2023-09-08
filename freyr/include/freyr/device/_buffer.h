#ifndef _FREYR_DEVICE_BUFFER_H
#define _FREYR_DEVICE_BUFFER_H

#include "detail.h"

namespace freyr::device {

    template<typename Derived, detail::backend_types>
    class base_buffer {
    public:
        using backend_type = detail::backend_types;

        base_buffer(size_t size, unsigned int usage_flags)
            : m_buffer_size(size), m_usage_flags(usage_flags) {}

        // Access the raw contents
        void *contents() { return static_cast<Derived *>(this)->contents_impl(); }

        // Set data
        template<typename T>
        void set_data(const std::vector<T> &data) {
            assert(m_buffer_size >= data.size() * sizeof(T));
            static_cast<Derived *>(this)->set_data_impl(data);
        }

        // Get data
        template<typename T>
        std::vector<T> get_data(size_t num_elements) {
            assert(m_buffer_size >= num_elements * sizeof(T));
            return static_cast<Derived *>(this)->get_data_impl(num_elements);
        }

        // Query size
        [[nodiscard]] size_t size() const { return m_buffer_size; }

        // Get usage flags
        [[nodiscard]] unsigned int usage_flags() const { return m_usage_flags; }

        // Synchronize buffer from device to host or host to device
        void synchronize() { static_cast<Derived *>(this)->synchronize_impl(); }

        // Copy data from host to this buffer
        template<typename T>
        void copy_from_host(const std::vector<T> &data) {
            assert(m_buffer_size >= data.size() * sizeof(T));
            static_cast<Derived *>(this)->copy_from_host_impl(data);
        }

        // Copy data from this buffer to host
        template<typename T>
        std::vector<T> copy_to_host(size_t num_elements) {
            assert(m_buffer_size >= num_elements * sizeof(T));
            return static_cast<Derived *>(this)->copy_to_host_impl(num_elements);
        }

    private:
        size_t       m_buffer_size;
        unsigned int m_usage_flags;// For example, read-only, write-only, read-write etc.
    };


}// namespace freyr::device


#endif//_FREYR_DEVICE_BUFFER_H
