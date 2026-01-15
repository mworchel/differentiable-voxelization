# Add the core library containing the algorithms
add_library(dvx INTERFACE common.hpp 
                          filter.hpp 
                          grid.hpp
                          math.hpp
                          winding_number.hpp)
target_include_directories(dvx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

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
  log.hpp log.cpp
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