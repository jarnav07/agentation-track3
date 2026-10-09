// Hand-written equivalent of snappy 1.1.10's CMake-generated config.h for x86-64 Linux / GCC.
// SNAPPY_HAVE_X86_CRC32 changes the match-finder hash and therefore the compressed bytes. It is
// pinned to 0 here (snappy.cc would turn it on under -msse4.2): that reproduces the codec bundled
// in the pyarrow 15.0.2 wheel byte for byte, and turning it on does not (tests/test_snappy.cpp).
#ifndef THIRD_PARTY_SNAPPY_OPENSOURCE_CMAKE_CONFIG_H_
#define THIRD_PARTY_SNAPPY_OPENSOURCE_CMAKE_CONFIG_H_
#define HAVE_ATTRIBUTE_ALWAYS_INLINE 1
#define HAVE_BUILTIN_CTZ 1
#define HAVE_BUILTIN_EXPECT 1
#define HAVE_FUNC_MMAP 1
#define HAVE_FUNC_SYSCONF 1
#define HAVE_LIBLZO2 0
#define HAVE_LIBZ 0
#define HAVE_LIBLZ4 0
#define HAVE_SYS_MMAN_H 1
#define HAVE_SYS_RESOURCE_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_UIO_H 1
#define HAVE_UNISTD_H 1
#define HAVE_WINDOWS_H 0
#define SNAPPY_HAVE_NEON 0
#define SNAPPY_HAVE_NEON_CRC32 0
#define SNAPPY_IS_BIG_ENDIAN 0
#define SNAPPY_HAVE_X86_CRC32 0
#define SNAPPY_HAVE_BMI2 0
#define SNAPPY_HAVE_SSSE3 0
#endif
