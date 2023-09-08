#ifndef FREYR_NDARRAY_FACTORY_H
#define FREYR_NDARRAY_FACTORY_H


namespace freyr::device {

    // linspace

    // mgrid
    template<typename T,
             typename Device = DEFAULT_DEVICE,
             typename Layout = storage_layout::row_major>
    std::tuple<ndarray<T, Device, Layout>, ndarray<T, Device, Layout>>
    mgrid(const std::vector<mgrid_args<T>> &params) {
        if (params.size() != 2) {
            throw std::invalid_argument("Only 2D grids are supported in this example.");
        }

        auto x = params[0];
        auto y = params[1];

        int x_steps = x.steps.value_or(static_cast<int>(x.end - x.start));
        int y_steps = y.steps.value_or(static_cast<int>(y.end - y.start));

        int total_elements = x_steps * y_steps;

        T x_step = (x.end - x.start) / (x_steps - 1);
        T y_step = (y.end - y.start) / (y_steps - 1);

        // Create flat storage for x and y using the ndarray class
        std::vector<T> x_storage(total_elements);
        std::vector<T> y_storage(total_elements);

        // Choose index calculation based on storage layout
        auto calc_index = std::is_same_v<Layout, storage_layout::row_major>
                                ? [](int i, int j, int y_steps) { return i * y_steps + j; }
                                : [](int i, int j, int y_steps) { return j * y_steps + i; };

        // Create a vector of indices from 0 to total_elements - 1
        std::vector<int> indices(total_elements);
        std::iota(indices.begin(), indices.end(), 0);

        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int index) {
            int i                       = index / y_steps;
            int j                       = index % y_steps;
            int calc_index_value        = calc_index(i, j, y_steps);
            x_storage[calc_index_value] = x.start + i * x_step;
            y_storage[calc_index_value] = y.start + j * y_step;
        });

        ndarray<T, Device, Layout> x_ndarray(std::move(x_storage));
        ndarray<T, Device, Layout> y_ndarray(std::move(y_storage));

        x_ndarray.set_dimensions({static_cast<size_t>(x_steps), static_cast<size_t>(y_steps)});
        y_ndarray.set_dimensions({static_cast<size_t>(x_steps), static_cast<size_t>(y_steps)});

        // Return the tuple using std::move to invoke move constructor
        return std::make_tuple(std::move(x_ndarray), std::move(y_ndarray));
    }
    // meshgrid

    // arange

    // zeros
    template<typename T,
             typename Device = DEFAULT_DEVICE,
             typename Layout = storage_layout::row_major>
    ndarray<T, Device, Layout> zeros(const std::vector<size_t> &shape) {
        // Calculate total size
        size_t total_size
                = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

        // Initialize storage with zeros
        //        std::vector<T> storage(total_size, T(0));

        std::fill(std::execution::par, storage.begin(), storage.end(), T(0));


        // Create an ndarray object
        ndarray<T, Device, Layout> array(storage);

        // Set the dimensions
        array.set_dimensions(shape);

        // Move data to device if needed
        array.to_device();

        return array;
    }

    // zeros_like
    template<typename T, typename Device, typename Layout>
    ndarray<T, Device, Layout> zeros_like(const ndarray<T, Device, Layout> &other) {
        // Get the shape of the other array
        std::vector<size_t> shape = other.shape();

        // Create a new ndarray initialized to zeros
        return zeros<T, Device, Layout>(shape);
    }

    // ones

    // ones_like

    // empty

    // empty_like

    // full

    // eye

    // identity

    // diag

    // tri

    // tril

    // triu

    // fromfile

    // frombuffer

    // copy


}// namespace freyr::device
#endif// FREYR_NDARRAY_FACTORY_H
