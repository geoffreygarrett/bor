#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


constexpr std::size_t array_length = 1 << 30;

template<typename Derived, std::size_t N>
class metal_op_base {
public:
    explicit metal_op_base(MTL::Device *device) : m_device(device), m_command_queue(nullptr) {}

    using buffer_array_type = std::array<MTL::Buffer *, N>;

    void init_with_device() {
        load_and_compile_shader();
        m_command_queue = m_device->newCommandQueue();
    }

    template<std::size_t Index, typename T>
    std::vector<T> fetch_data_from_buffer() {
        static_assert(Index < N, "Index out of range");
        auto* buffer_ptr = m_buffers[Index]->contents();
        std::vector<T> host_data(array_length);
        std::memcpy(host_data.data(), buffer_ptr, array_length * sizeof(T));
        return host_data;
    }

    void load_and_compile_shader() {
        NS::Error *error = nullptr;

        const std::string shader_path   = static_cast<Derived *>(this)->get_shader_file_path();
        std::string       shader_source = load_shader_source(shader_path);

        auto library_string = NS::String::string(shader_source.c_str(), NS::ASCIIStringEncoding);
        auto library        = m_device->newLibrary(library_string, &error);

        if (!library || error) {
            std::cerr << "Failed to compile shader from source. Error: "
                      << error->localizedDescription();
            std::exit(-1);
        }

        const std::string function_name = static_cast<Derived *>(this)->get_shader_function_name();
        auto ns_string = NS::String::string(function_name.c_str(), NS::ASCIIStringEncoding);
        auto function  = library->newFunction(ns_string);

        if (!function) {
            std::cerr << "Failed to find the function: " << function_name;
            std::exit(-1);
        }

        m_add_function_pso = m_device->newComputePipelineState(function, &error);
        if (error) {
            std::cerr << "Failed to create pipeline state. Error: "
                      << error->localizedDescription();
            std::exit(-1);
        }
    }

    std::string load_shader_source(const std::string &path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << path;
            std::exit(-1);
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    void execute() { static_cast<Derived *>(this)->execute_impl(); }

    template<std::size_t Index>
    auto get_buffer() -> MTL::Buffer * {
        static_assert(Index < N, "Index out of range");
        return m_buffers[Index];
    }

//    // to host
//    template<std::size_t Index>
//    auto buffer_to_host() -> void * {
//        static_assert(Index < N, "Index out of range");
//        return m_buffers[Index]->contents();
//    }

    template<std::size_t Index, typename T>
    void set_buffer_data(const std::vector<T>& data) {
        static_assert(Index < N, "Index out of range");
        m_buffers[Index] = prepare_buffer(data);
    }

    template<std::size_t Index, typename T>
    void update_buffer_data(const std::vector<T>& data) {
        static_assert(Index < N, "Index out of range");
        auto* buffer_ptr = m_buffers[Index]->contents();
        std::memcpy(buffer_ptr, data.data(), data.size() * sizeof(T));
    }

    template<typename... Buffers>
    void prepare_data(Buffers... data_buffers) {
        m_buffers = {prepare_buffer(data_buffers)...};
    }

protected:
    template<typename T>
    auto prepare_buffer(const std::vector<T> &data) -> MTL::Buffer * {
        auto buffer = m_device->newBuffer(data.size() * sizeof(T), MTL::ResourceStorageModeShared);
        std::memcpy(buffer->contents(), data.data(), data.size() * sizeof(T));
        return buffer;
    }

//    template<typename... Buffers>
//    void prepare_data(Buffers... data_buffers) {
//        m_buffers = {prepare_buffer(data_buffers)...};
//    }

    MTL::Device               *m_device;
    MTL::CommandQueue         *m_command_queue;
    MTL::ComputePipelineState *m_add_function_pso;

    //    buffer_tuple_type m_buffers;
    buffer_array_type m_buffers;

    //    std::tuple<MTL::Buffer*, MTL::Buffer*> m_buffers;
};
