#ifndef FREYR_CORE_H
#define FREYR_CORE_H

#include <functional>
#include <type_traits>
#include <variant>
#include <vector>

namespace freyr {

//#define CRTP_VAR_CALL(variant, method, ...)                                                       \
//    std::visit([&](auto &instance) -> decltype(auto) { return instance.method(__VA_ARGS__); },    \
//               variant)
#define CRTP_VAR_CALL(variant, method)                                                            \
    [&](auto &&...args) -> decltype(auto) {                                                       \
        return std::visit(                                                                        \
                [&](auto &instance) -> decltype(auto) {                                           \
                    return instance.method(std::forward<decltype(args)>(args)...);                \
                },                                                                                \
                variant);                                                                         \
    }


}// namespace freyr

#endif// FREYR_CORE_H