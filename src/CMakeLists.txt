# Add the core library containing the algorithms
add_library(dvx allocator.hpp allocator.cpp
                bitset.hpp bitset.cpp
                closed_form.hpp
                common.hpp 
                coverage.hpp
                differentiation.hpp
                filter.hpp 
                flags.hpp
                grid.hpp
                log.hpp log.cpp
                math.hpp
                monte_carlo.hpp
                winding_number.hpp)
target_include_directories(dvx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(dvx PUBLIC DVX_VERSION="${PROJECT_VERSION}")

# Add the Python extension
nanobind_add_module(
  dvx_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ext.cpp
)

target_link_libraries(dvx_ext PRIVATE dvx)

# Install directive for scikit-build-core
install(TARGETS dvx_ext LIBRARY DESTINATION .)

# Add tests and playgrounds
if (DVX_BUILD_TESTS)
    message(STATUS "Building tests")
else()
    message(STATUS "Skipping tests")
endif()