// Fast parquet writer for the two fixed output schemas, producing the same bytes as
// parquet::arrow::WriteTable with pyarrow.parquet.write_table's defaults (see pqfast.cpp).
#pragma once
#include <arrow/api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastsim {

struct PqColumn {
  enum Kind { I64, I32, STR } kind;
  const int64_t* i64 = nullptr;
  const int32_t* i32 = nullptr;
  const int32_t* codes = nullptr;  // STR: index into names
  const char* const* names = nullptr;
  int n_names = 0;
  const uint8_t* valid = nullptr;  // 1 = valid; nullptr = no nulls
};

// Serialises a table of `n` rows with `schema` (all fields nullable, as in writer.cpp) into a file
// image, encoding column chunks on up to `threads` threads. Returns false with `err` set if anything
// is unexpected; the caller then uses libparquet.
bool pq_write_fast(const std::shared_ptr<arrow::Schema>& schema, const std::vector<PqColumn>& cols, int64_t n,
                   std::shared_ptr<arrow::Buffer>* out, std::string& err, int threads = 1);

}  // namespace fastsim
