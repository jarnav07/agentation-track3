// trace.parquet / message_trace.parquet through Arrow C++ / libparquet — the very libraries the
// pyarrow 15.0.2 wheel ships (linked from its site-packages directory) — with the writer and
// Arrow-writer properties that pyarrow.parquet.write_table uses by default, and the same pandas
// schema-metadata string the baseline's DataFrame.to_parquet stores. Same library, same options,
// same metadata → the same bytes (checked by scripts/check_identical.py).
#pragma once
#include <string>

#include "engine.h"

namespace fastsim {

// `sha` (optional) receives the SHA-256 hex digest of the written file.
bool write_trace_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);
bool write_ledger_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);

}  // namespace fastsim
