
#include <iostream>
#include <op_addition.h>
//#include <op_subtraction.h>
#include <vector>

#include <Eigen/Core>
#include <freyr/device/device.h>

using namespace freyr;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Prepare data (assuming you have data populated in A, B, and C)
    device::ndarray A(array_length, 1.0f);// Use the constructor for std::vector
    device::ndarray B(array_length, 2.0f);// Use the constructor for length and initial value
    device::ndarray C(array_length, 0.0f);// Use the constructor for length and initial value

    // Assign the values of A and B as the index of the array
    for (unsigned long i = 0; i < array_length; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }

    // Copy to device
    A.to_device();
    B.to_device();
    C.to_device();

    auto height = 1080;
    auto width  = 1920;

    // create a meshgrid
    auto [uu, vv] = device::mgrid<int>({
            {300, height},
            {300,  width}
    });

    // iterate and print values
    std::cout << "uu.size() = " << uu.size() << std::endl;
    std::cout << "vv.size() = " << vv.size() << std::endl;
    std::cout << "uu.shape() = " << uu.shape() << std::endl;

    // print the values at 0,0 and 1079,1919
    std::cout << "uv[0,0] = (" << uu[0] << ", " << vv[0] << ")" << std::endl;
    std::cout << "uv[1079,1919] = (" << uu.at(1079, 1919) << ", " << vv.at(1079, 1919) << ")"
              << std::endl;

    // copy to device
//    uu.to_device();
//    vv.to_device();

    auto ww = device::zeros_like(uu);
    std::cout<< uu[2000] << std::endl;
    std::cout<< uu.data()[3000] << std::endl;

    // Get the default device
    auto *device = device::manager<>::get_default_device();

    // Initialize Metal device
    NS::AutoreleasePool *p_pool = NS::AutoreleasePool::alloc()->init();

    // Create an instance of the addition operator and initialize it
    op_addition add_op(device);
    add_op.init_with_device();

    // Using set_buffer_data method to populate Metal buffers directly
    add_op.set_buffer_data<0, int>(uu.data());
    add_op.set_buffer_data<1, int>(vv.data());
    add_op.set_buffer_data<2, int>(ww.data());

    //    // Execute the operation (assuming execute method will use the prepared buffers)
    add_op.execute();

    // Assuming the data in the buffer is of type float
    auto result_ptr = (int *) add_op.get_buffer<2>()->contents();

    // Now you can use typed_pointer as an array of floats
    for (unsigned long i = 0; i < uu.size(); i++) { std::cout << result_ptr[i] << std::endl; }

    std::cout << "Done!" << std::endl;

    //
    //    scene.add_object(new sphere({0, 0, 0}, 1.0f));
    //
    //    //
    //    camera = new camera::perspective({0, 0, 0}, {0, 0, 1}, {0, 1, 0}, 90, 1.0f);
    //    backend.shader("raytrace", "raytrace.metal");


    // Using set_buffer_data method to populate Metal buffers directly

    //    // Using set_buffer_data method to populate Metal buffers directly
    //    add_op.set_buffer_data<0, float>(A);
    //    add_op.set_buffer_data<1, float>(B);
    //    add_op.set_buffer_data<2, float>(C);

    //    //    // Execute the operation (assuming execute method will use the prepared buffers)
    //    add_op.execute();
    //
    //    // Assuming the data in the buffer is of type float
    //    auto result_ptr = (float *) add_op.get_buffer<2>()->contents();
    //
    //    // Now you can use typed_pointer as an array of floats
    //    for (unsigned long i = 0; i < array_length; i++) { std::cout << result_ptr[i] << std::endl; }
    //
    //    std::cout << "Done!" << std::endl;
    //    return 0;
}
