#pragma once
#include "op_base.h"
#include "op_addition.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


using op_type = std::variant<op_addition>;

//
//class chain_metal_op {
//public:
//    MTL::Device         *m_device;
//    MTL::CommandQueue   *m_command_queue;
//    std::vector<op_type> m_metal_ops;
//
//    chain_metal_op() {
//        // Prepare command buffer and encoder
//        MTL::CommandBuffer         *command_buffer  = m_command_queue->commandBuffer();
//        MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();
//    }
//
//    void add_metal_op(metal_op_base *op) { m_metal_ops.push_back(op); }
//
//    void init_with_device() {
//        for (auto &op: m_metal_ops) { op->init_with_device(); }
//    }
//
//    void send_compute_command() {
//        for (auto &op: m_metal_ops) { op->send_compute_command(); }
//    }
//
//    void wait_until_completed() {
//        for (auto &op: m_metal_ops) { op->wait_until_completed(); }
//    }
//
//    void release() {
//        for (auto &op: m_metal_ops) { op->release(); }
//    }
//};
