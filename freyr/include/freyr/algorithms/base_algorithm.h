#ifndef BASE_ALGORITHM_H
#define BASE_ALGORITHM_H

namespace freyr {

    template<typename Backend, typename... Args>
    struct Algorithm {
        static void compute(Args... args) {
            Backend::compute(args...);
        }
    };

}// namespace freyr

#endif// BASE_ALGORITHM_H