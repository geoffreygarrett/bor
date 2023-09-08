
#pragma once
#include "op_base.h"

constexpr unsigned int array_length = 1024;


class op_subtraction : public metal_op_base<op_addition, 3> {
public:
    using metal_op_base<op_subtraction, 3>::metal_op_base;

    explicit op_subtraction(MTL::Device *device) : metal_op_base<op_subtraction, 3>(device) {
        // Initialize buffers
        prepare_data(std::vector<float>(array_length),
                     std::vector<float>(array_length),
                     std::vector<float>(array_length));
    }

    static constexpr std::string get_shader_file_path() {
        return "freyr/assets/shaders/metal/metal_shaders.metallib";
    }

    static constexpr std::string get_shader_function_name() { return "op_subtraction"; }

    void execute_impl() {
        // Prepare command buffer and encoder
        MTL::CommandBuffer         *command_buffer  = m_command_queue->commandBuffer();
        MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();

        // Set the pipeline state and buffers
        compute_encoder->setComputePipelineState(m_add_function_pso);
        compute_encoder->setBuffer(get_buffer<0>(), 0, 0);
        compute_encoder->setBuffer(get_buffer<1>(), 0, 1);
        compute_encoder->setBuffer(get_buffer<2>(), 0, 2);

        // Grid and thread configuration
        MTL::Size    grid_size          = MTL::Size(array_length, 1, 1);
        NS::UInteger thread_group_size_ = m_add_function_pso->maxTotalThreadsPerThreadgroup();
        if (thread_group_size_ > array_length) { thread_group_size_ = array_length; }
        MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

        // Dispatch threads
        compute_encoder->dispatchThreads(grid_size, thread_group_size);
        compute_encoder->endEncoding();

        // Commit and wait for completion
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
    }
};
