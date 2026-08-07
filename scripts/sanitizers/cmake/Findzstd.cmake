# Findzstd.cmake -- minimal module-mode shim for the nightly sanitizer gate.
#
# The upstream rocjitsu build (ROCm/rocm-systems) does `find_package(zstd
# REQUIRED)` expecting zstd's *config* package (zstdConfig.cmake + the
# zstd::libzstd_shared / zstd::libzstd_static targets). Ubuntu's libzstd-dev
# (the rocm/pytorch CI base) ships the headers and library but no CMake package
# config, and rocjitsu's own cmake/ dir carries no Findzstd fallback, so the
# configure aborts. Placing this file on CMAKE_MODULE_PATH lets find_package
# resolve in module mode and hand rocjitsu the zstd::libzstd_shared target it
# aliases from -- wrapping whatever libzstd the distro package installed.
include(FindPackageHandleStandardArgs)

find_path(zstd_INCLUDE_DIR NAMES zstd.h)
find_library(zstd_LIBRARY NAMES zstd)

find_package_handle_standard_args(zstd
  REQUIRED_VARS zstd_LIBRARY zstd_INCLUDE_DIR
)

if(zstd_FOUND AND NOT TARGET zstd::libzstd_shared)
  add_library(zstd::libzstd_shared UNKNOWN IMPORTED)
  set_target_properties(zstd::libzstd_shared PROPERTIES
    IMPORTED_LOCATION "${zstd_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(zstd_INCLUDE_DIR zstd_LIBRARY)
