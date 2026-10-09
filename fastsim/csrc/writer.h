// trace.parquet / message_trace.parquet, byte-identical to what the baseline's
// DataFrame.to_parquet (pyarrow 15.0.2 / parquet-cpp, write_table defaults, pandas schema metadata)
// writes, through the self-contained writer in pqfast.cpp (checked by scripts/check_identical.py).
#pragma once
#include <string>

#include "engine.h"
#include "pqfast.h"

namespace fastsim {

// Encoded files (false: the writer declined this output; the caller falls back to Python). The
// ledger is read from o.lrows when o.ledger_rows is set, else from the l_* columns. Write them
// with PqImage::write_file.
bool trace_image(const Output& o, PqImage* img, int threads);
bool ledger_image(const Output& o, PqImage* img, int threads);

// Starts encoding the ledger from the rows the engine publishes in `feed` (Output::ledger_feed),
// on `threads` threads; PqStream::finish then yields the image once the run has ended.
PqStream* ledger_stream(const RowFeed& feed, int threads);

// Encode with one thread and write. `sha` (optional) receives the SHA-256 hex digest of the written file.
bool write_trace_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);
bool write_ledger_parquet(const Output& o, const std::string& path, std::string& err, std::string* sha = nullptr);

}  // namespace fastsim
