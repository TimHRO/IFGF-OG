# Internal static library: octree and interaction lists (the engine itself is header-only, plan/face_plan.hpp)
add_library(ifgf STATIC
  tree/tree.cpp
  lists/lists.cpp
)
add_library(ifgf::ifgf ALIAS ifgf)

target_include_directories(ifgf PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_compile_features(ifgf PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(ifgf PUBLIC ifgf_options Threads::Threads)

# --- Library API (ifgf_library.hpp): shared library for callers without SYCL ------------------------------------
if(IFGF_BUILD_LIBRARY)
  set(IFGF_LIBRARY_SOURCES api/library.cpp api/engine_sl.cpp api/engine_dl.cpp api/engine_cf.cpp)
  add_library(ifgf_library SHARED ${IFGF_LIBRARY_SOURCES})
  add_library(ifgf::ifgf_library ALIAS ifgf_library)
  target_link_libraries(ifgf_library PRIVATE ifgf)
  add_sycl_to_target(TARGET ifgf_library SOURCES ${IFGF_LIBRARY_SOURCES})
  target_compile_options(ifgf_library PRIVATE -Wno-pass-failed)
  # public interface: only the plain C++17 header ifgf/ifgf_library.hpp
  target_include_directories(ifgf_library INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(ifgf_library INTERFACE cxx_std_17)
  set_target_properties(ifgf_library PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    INSTALL_RPATH_USE_LINK_PATH ON)  # finds the AdaptiveCpp runtime after installation
  # export only the API (the internal static library is linked in, its symbols stay hidden)
  target_link_options(ifgf_library PRIVATE "LINKER:--exclude-libs,ALL")

  install(TARGETS ifgf_library EXPORT ifgfTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES ifgf_library.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ifgf)
  install(EXPORT ifgfTargets NAMESPACE ifgf:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/ifgfConfig.cmake.in
    ${PROJECT_BINARY_DIR}/ifgfConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf)
  write_basic_package_version_file(${PROJECT_BINARY_DIR}/ifgfConfigVersion.cmake COMPATIBILITY SameMinorVersion)
  install(FILES ${PROJECT_BINARY_DIR}/ifgfConfig.cmake ${PROJECT_BINARY_DIR}/ifgfConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ifgf)
endif()
