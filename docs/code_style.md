# Code Style

This file is the starting point to the code styling used across the BOR project.
This file is necessitated by the vast differences across recommended styling for
C++, Python and commonalities across languages and industries. This document serves
to attempt to create a homogenous styling across all languages in the BOR project,
while motivating and selecting standards according to the authors involved fields
and industries.

- C++
    - Templates shall be concise single letters, T, U, V, etc, with the exception of
      size information, e.g. N, M, K, etc.
        - T is generally reserved for a primary container type that is utilised by
          the class, e.g. `T = std::vector<U>`.
        - U is generally reserved for the primary scalar type, e.g. `U = double`.
        - T and U are ordered such, as often U can be deduced by the specified T,
          leading to more concise, and equally expressive code.
        - In order to be expressive in meaning of the type, generally, the first
          order of business in the public interface of a class is to assign type
          aliases to the template parameters, e.g. `using scalar_type = U;`.
        - We avoid the use of `<type>_t`, although it may be more idiomatic, it
          has the potential to clash with POSIX types which reserves this suffix,
          e.g. `size_t`.
        - An alternative could be `<type>_tt`, however practice is not as popular as
          the alternative solution to this problem, which is to use the `<type>_type`.
          Therefore in order to balance conciseness, expressiveness and immediate clarity
          to the largest audience, we opt for the latter. Therefore the following shall
          be a repeating pattern throughout the C++ code base:

          ``` C++
            template <typename T, typename U>
            class Foo {
            public:
                using scalar_type = U;
                using container_type = T;
          
                // ...
          
                scalar_type bar() const;
                container_type baz() const;
            };
          ```

The prefix `generate_<type>` is used to denote a function that generates a type,
which does not necessarily return a deterministic value, e.g. `generate_random_vector`,
`generate_orthogonal_vector`, etc.

## Math is Math and Code is Code, but they're not exclusive

Math is math and code is code, however not all math can be expressed in words, and
therefore the author does not shy away from allowing the use of concise mathematical
notation in code, e.g. `x = A * y + b`, `x = A * y + b * z`, etc. This is especially
necessary in order to allow for readable code in the case of linear algebra, where
the use of mathematical notation is ubiquitous. Attempting to go against the grain
in this case would result in code that is less readable, and therefore less maintainable.
However, all mathematical notation must be accompanied by a code comment block, within
its scope, that explains the mathematical notation in words, e.g.

``` C++
    // Nomenclature:
    // =================
    // n : unit normal vector of plane
    // ...
    
    // Ensure that the normal vector is normalized
    auto n = normal.normalized();
```

Another reason is that motivates the need to address this in detail is the fact that
there are vast occurrences of condense sets of equations in Mimir that, if written in words, would be
unreadable, unmaintainable, unmanageable, and therefore unusable. Furthermore, not
every developer is a mathematician, and therefore the use of mathematical notation
is necessary to allow for the use of the code by a wider audience.