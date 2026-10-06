// trace.parquet / message_trace.parquet through Arrow C++ / libparquet — the very libraries the
// pyarrow 15.0.2 wheel ships (linked from its site-packages directory) — with the writer and
// Arrow-writer properties that pyarrow.parquet.write_table uses by default, and the same pandas
// schema-metadata string the baseline's DataFrame.to_parquet stores. Same library, same options,
// same metadata → the same bytes (checked by scripts/check_identical.py).
#pragma once
#include <arrow/buffer.h>

#include <memory>
#include <string>

#include "engine.h"

namespace fastsim {

// Threads the fast writer may use to encode column chunks (default 1).
void set_writer_threads(int n);

// File images from the fast writer (false: not available for this output; use write_*_parquet).
bool trace_image(const Output& o, std::shared_ptr<arrow::Buffer>* img);
bool ledger_image(const Output& o, std::shared_ptr<arrow::Buffer>* img);
// Writes an image to `path` and hashes it (SHA-256 hex into `sha`).
bool save_image(const std::shared_ptr<arrow::Buffer>& img, const std::string& path, std::string* sha, std::string& err);

// `sha` (optional) receives the SHA-256 hex digest of the written file.
bool write_trace_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);
bool write_ledger_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);

}  // namespace fastsim
