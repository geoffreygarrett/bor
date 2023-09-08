// Assume MTL and Eigen namespaces are already declared

// Shape struct for holding tensor dimensions
struct Shape {
    int width;
    int height;
    int depth;
};

template<typename T>
class device_array {
public:
    // Constructor specialization for Eigen matrix
    template<typename U>
    device_array(MTL::Device *device,
                 U          &&data,
                 typename std::enable_if<is_eigen_matrix<U>::value>::type * = nullptr)
        : shape{data.cols(), data.rows(), 1} {
        // Initialize from Eigen matrix
    }

    // Constructor specialization for std::vector
    template<typename U>
    DeviceArray(MTL::Device *device,
                U          &&data,
                typename std::enable_if<!is_eigen_matrix<U>::value>::type * = nullptr)
        : shape{static_cast<int>(data.size()), 1, 1} {
        // Initialize from std::vector or native array
    }

    // Handle 3D tensors
    DeviceArray(MTL::Device *device, const std::vector<std::vector<std::vector<T>>> &tensor)
        : shape{static_cast<int>(tensor[0][0].size()),
                static_cast<int>(tensor[0].size()),
                static_cast<int>(tensor.size())} {
        // Initialize for 3D tensor
    }

    // Additional overloads for matrix and tensor operations
    DeviceArray<T> &matmul(const DeviceArray<T> &rhs);

    // Transpose for matrices
    DeviceArray<T> transpose();

    void           execute();
    std::vector<T> to_host();
    // ... other methods and operators ...

private:
    Shape                              shape;
    MTL::Buffer                       *m_buffer;
    std::vector<std::function<void()>> m_operations;

    void execute_single_operation(const std::function<void()> &op);
};

// Implement matrix multiplication
template<typename T>
DeviceArray<T> &DeviceArray<T>::matmul(const DeviceArray<T> &rhs) {
    m_operations.push_back([this, &rhs]() {
        // Your Metal API calls for matrix multiplication
    });
    return *this;
}

// Implement transpose
template<typename T>
DeviceArray<T> DeviceArray<T>::transpose() {
    // Add a transpose operation to the queue
    m_operations.push_back([this]() {
        // Your Metal API calls for transpose
    });
    return *this;
}
