// fastsim-native: the `simulate` / `simulate-batch` verbs without a Python interpreter.
//
// Same engine (engine.cpp), same outputs, but scenario parsing, seeding and parquet writing are
// native too, so a run costs a few tens of milliseconds of process start-up instead of the
// interpreter + NumPy + pyarrow imports (~0.2-0.3 s), which counts in Final's container-window
// timing. Anything the native setup is not certain about (see native_setup.cpp) is handed, whole,
// to the Python implementation (`python3 -m fastsim.simulate ...`), which is byte-identical by the
// same oracle; so is any engine error, so that error behaviour stays the Python path's.
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "engine.h"
#include "json.h"
#include "native_setup.h"
#include "sha256.h"
#include "writer.h"

using namespace fastsim;

namespace {

char** g_argv = nullptr;

[[noreturn]] void fallback(const char* module, const std::string& why) {
  if (std::getenv("FASTSIM_VERBOSE")) std::fprintf(stderr, "fastsim-native: falling back to Python (%s)\n", why.c_str());
  if (std::getenv("FASTSIM_NO_FALLBACK")) {
    std::fprintf(stderr, "fastsim-native: fallback disabled: %s\n", why.c_str());
    std::exit(3);
  }
  const char* py = std::getenv("FASTSIM_PYTHON");
  if (!py) py = "python3";
  std::vector<char*> args;
  args.push_back((char*)py);
  args.push_back((char*)"-m");
  args.push_back((char*)module);
  for (char** a = g_argv + 1; *a; a++) args.push_back(*a);
  args.push_back(nullptr);
  execvp(py, args.data());
  std::perror("fastsim-native: exec python");
  std::exit(127);
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

void mkdirs(const std::string& dir) {
  std::string cur;
  for (size_t i = 0; i < dir.size(); i++) {
    cur += dir[i];
    if (dir[i] == '/' && cur.size() > 1) mkdir(cur.c_str(), 0777);
  }
  mkdir(dir.c_str(), 0777);
}

std::string parent_of(const std::string& p) {
  size_t k = p.find_last_of('/');
  if (k == std::string::npos) return ".";
  if (k == 0) return "/";
  return p.substr(0, k);
}

// Python repr(float)
std::string py_repr(double x) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
  std::string s(buf, r.ptr);
  bool neg = s[0] == '-';
  if (neg) s.erase(0, 1);
  size_t e = s.find('e');
  std::string mant = s.substr(0, e);
  int exp10 = std::atoi(s.c_str() + e + 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits += c;
  std::string out;
  if (exp10 >= -4 && exp10 < 16) {
    int point = exp10 + 1;  // digits before the decimal point
    if (point <= 0) {
      out = "0." + std::string((size_t)(-point), '0') + digits;
    } else if ((size_t)point >= digits.size()) {
      out = digits + std::string((size_t)point - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, (size_t)point) + "." + digits.substr((size_t)point);
    }
  } else {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    out += eb;
  }
  return neg ? "-" + out : out;
}

bool ascii_printable(const std::string& s) {
  for (unsigned char c : s)
    if (c < 0x20 || c >= 0x7f || c == '"' || c == '\\') return false;
  return true;
}

int64_t peak_rss_bytes() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return (int64_t)ru.ru_maxrss * 1024;
}

struct RunResult {
  std::string scenario_id;
  int64_t seed = 0;
  int64_t n_events = 0, n_messages = 0;
  double wall = 0.0;
  std::string trace_sha, msg_sha;
  std::string error;
};

std::string events_json(const RunResult& r) {
  double eps = r.wall > 0 ? (double)r.n_events / r.wall : 0.0;
  std::string s = "{\n";
  s += "  \"scenario_id\": \"" + r.scenario_id + "\",\n";
  s += "  \"seed\": " + std::to_string(r.seed) + ",\n";
  s += "  \"n_events\": " + std::to_string(r.n_events) + ",\n";
  s += "  \"wall_clock_sec\": " + py_repr(r.wall) + ",\n";
  s += "  \"events_per_sec\": " + py_repr(eps) + ",\n";
  s += "  \"trace_sha256\": \"" + r.trace_sha + "\",\n";
  s += "  \"n_messages\": " + std::to_string(r.n_messages) + ",\n";
  s += "  \"message_trace_sha256\": \"" + r.msg_sha + "\",\n";
  s += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
  s += "  \"gpu_seconds\": 0.0\n}\n";
  return s;
}

// Parse + build the spec; false = needs the Python path.
bool prepare(const std::string& config_path, const char* seed_override, Spec& spec, ScenarioMeta& meta,
             std::string& why) {
  std::string text;
  if (!read_file(config_path, text)) { why = "cannot read " + config_path; return false; }
  json::Value sc;
  if (!json::parse(text, sc)) { why = "JSON the native parser does not accept"; return false; }
  if (seed_override) {
    char* end = nullptr;
    long long v = std::strtoll(seed_override, &end, 10);
    if (!end || *end) { why = "bad --seed"; return false; }
    json::Value sv;
    sv.type = json::INT;
    sv.i = v;
    bool set = false;
    if (sc.type == json::OBJECT)
      for (auto& kv : sc.obj)
        if (kv.first == "seed") { kv.second = sv; set = true; }
    if (!set && sc.type == json::OBJECT) sc.obj.emplace_back("seed", sv);
  }
  if (!build_native_spec(sc, spec, meta, why)) return false;
  if (!ascii_printable(meta.scenario_id)) { why = "non-ASCII scenario_id"; return false; }
  return true;
}

// Run + write trace, ledger, events.json into out_dir.
bool run_one(Spec& spec, const ScenarioMeta& meta, const std::string& trace_path, RunResult& r) {
  r.scenario_id = meta.scenario_id;
  r.seed = meta.seed;
  Output out;
  auto t0 = std::chrono::steady_clock::now();
  bool ok = run_engine(spec, out);
  auto t1 = std::chrono::steady_clock::now();
  if (!ok) { r.error = "engine: " + out.error; return false; }
  r.wall = std::chrono::duration<double>(t1 - t0).count();
  r.n_events = (int64_t)out.t_ns.size();
  r.n_messages = (int64_t)out.l_t_recv.size();
  std::string dir = parent_of(trace_path);
  mkdirs(dir);
  std::string msg_path = dir + "/message_trace.parquet";
  std::string err;
  // The two files are independent: write + hash them concurrently.
  auto t2 = std::chrono::steady_clock::now();
  std::string err2;
  bool ok_ledger = true;
  std::thread ledger_thread([&] {
    ok_ledger = write_ledger_parquet(out, msg_path, err2);
    if (ok_ledger) r.msg_sha = sha256_file(msg_path);
  });
  bool ok_trace = write_trace_parquet(out, trace_path, err);
  if (ok_trace) r.trace_sha = sha256_file(trace_path);
  ledger_thread.join();
  if (!ok_trace || !ok_ledger) { r.error = "write: " + err + err2; return false; }
  auto t3 = std::chrono::steady_clock::now();
  if (std::getenv("FASTSIM_TIMING")) {
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr, "fastsim-native timing: engine %.1f ms, write+sha %.1f ms\n", ms(t0, t1), ms(t2, t3));
  }
  std::string ev = events_json(r);
  FILE* f = std::fopen((dir + "/events.json").c_str(), "wb");
  if (!f) { r.error = "cannot write events.json"; return false; }
  std::fwrite(ev.data(), 1, ev.size(), f);
  std::fclose(f);
  return true;
}

int available_cpus() {
  cpu_set_t set;
  int cpus = 1;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) cpus = CPU_COUNT(&set);
  std::string txt;
  if (read_file("/sys/fs/cgroup/cpu.max", txt)) {
    long long q = 0, p = 0;
    char qs[32] = {0};
    if (std::sscanf(txt.c_str(), "%31s %lld", qs, &p) == 2 && std::strcmp(qs, "max") != 0 && p > 0) {
      q = std::atoll(qs);
      cpus = std::min<int>(cpus, std::max<int>(1, (int)(q / p)));
    }
  } else {
    std::string qt, pt;
    if (read_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", qt) && read_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us", pt)) {
      long long q = std::atoll(qt.c_str()), p = std::atoll(pt.c_str());
      if (q > 0 && p > 0) cpus = std::min<int>(cpus, std::max<int>(1, (int)(q / p)));
    }
  }
  if (const char* env = std::getenv("FASTSIM_BATCH_WORKERS")) cpus = std::max(1, std::atoi(env));
  return std::max(1, cpus);
}

int cmd_simulate(int argc, char** argv) {
  const char* config = nullptr;
  const char* out = nullptr;
  const char* seed = nullptr;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "simulate" && i == 1) continue;
    if (a == "--config" && i + 1 < argc) config = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--seed" && i + 1 < argc) seed = argv[++i];
    else fallback("fastsim.simulate", "unrecognised argument " + a);
  }
  if (!config || !out) fallback("fastsim.simulate", "missing --config/--out");
  Spec spec;
  ScenarioMeta meta;
  std::string why;
  if (!prepare(config, seed, spec, meta, why)) fallback("fastsim.simulate", why);
  RunResult r;
  if (!run_one(spec, meta, out, r)) fallback("fastsim.simulate", r.error);
  std::string dir = parent_of(out);
  std::string ev;
  read_file(dir + "/events.json", ev);
  // stdout: one-line JSON, like the Python verb
  std::string line;
  for (char c : ev)
    if (c != '\n') line += c;
  std::printf("%s\n", line.c_str());
  return 0;
}

int cmd_simulate_batch(int argc, char** argv) {
  const char* batch_dir = nullptr;
  const char* out_dir = nullptr;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "simulate-batch" && i == 1) continue;
    if (a == "--batch-dir" && i + 1 < argc) batch_dir = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) out_dir = argv[++i];
    else fallback("fastsim.simulate_batch", "unrecognised argument " + a);
  }
  if (!batch_dir || !out_dir) fallback("fastsim.simulate_batch", "missing --batch-dir/--out-dir");
  // sorted(Path(batch_dir).glob("*.json")) — glob skips dot-files
  std::vector<std::string> names;
  DIR* d = opendir(batch_dir);
  if (!d) fallback("fastsim.simulate_batch", "cannot open batch dir");
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 5 && n[0] != '.' && n.compare(n.size() - 5, 5, ".json") == 0) {
      struct stat st;
      std::string full = std::string(batch_dir) + "/" + n;
      if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) names.push_back(n);
    }
  }
  closedir(d);
  std::sort(names.begin(), names.end());
  if (names.empty()) fallback("fastsim.simulate_batch", "no sub-scenarios");

  const size_t n = names.size();
  std::vector<Spec> specs(n);
  std::vector<ScenarioMeta> metas(n);
  for (size_t i = 0; i < n; i++) {
    std::string why;
    if (!prepare(std::string(batch_dir) + "/" + names[i], nullptr, specs[i], metas[i], why))
      fallback("fastsim.simulate_batch", names[i] + ": " + why);
  }
  std::vector<RunResult> results(n);
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  const int workers = std::min<int>(available_cpus(), (int)n);
  auto t0 = std::chrono::steady_clock::now();
  auto work = [&]() {
    for (size_t i; (i = next.fetch_add(1)) < n;) {
      std::string stem = names[i].substr(0, names[i].size() - 5);
      if (!run_one(specs[i], metas[i], std::string(out_dir) + "/" + stem + "/trace.parquet", results[i]))
        failed = true;
    }
  };
  if (workers <= 1) {
    work();
  } else {
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; w++) pool.emplace_back(work);
    for (auto& t : pool) t.join();
  }
  auto t1 = std::chrono::steady_clock::now();
  if (failed) fallback("fastsim.simulate_batch", "a sub-scenario failed natively");
  double wall = std::chrono::duration<double>(t1 - t0).count();
  int64_t total = 0;
  for (auto& r : results) total += r.n_events;
  std::string s = "{\n";
  s += "  \"n_scenarios\": " + std::to_string(n) + ",\n";
  s += "  \"total_events\": " + std::to_string(total) + ",\n";
  s += "  \"wall_clock_sec\": " + py_repr(wall) + ",\n";
  s += "  \"events_per_sec\": " + py_repr(wall > 0 ? (double)total / wall : 0.0) + ",\n";
  s += "  \"peak_memory_bytes\": " + std::to_string(peak_rss_bytes()) + ",\n";
  s += "  \"gpu_seconds\": 0.0,\n";
  s += "  \"per_scenario\": [";
  for (size_t i = 0; i < n; i++) {
    std::string stem = names[i].substr(0, names[i].size() - 5);
    s += i ? ",\n" : "\n";
    s += "    {\n      \"sub\": \"" + stem + "\",\n      \"n_events\": " + std::to_string(results[i].n_events) +
         ",\n      \"trace_sha256\": \"" + results[i].trace_sha + "\"\n    }";
  }
  s += "\n  ]\n}\n";
  mkdirs(out_dir);
  FILE* f = std::fopen((std::string(out_dir) + "/batch_events.json").c_str(), "wb");
  if (!f) { std::perror("batch_events.json"); return 1; }
  std::fwrite(s.data(), 1, s.size(), f);
  std::fclose(f);
  std::string line;
  for (char c : s)
    if (c != '\n') line += c;
  std::printf("%s\n", line.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  g_argv = argv;
  std::string prog = argv[0];
  std::string base = prog.substr(prog.find_last_of('/') + 1);
  std::string verb = argc > 1 ? argv[1] : "";
  bool batch = base == "simulate-batch" || verb == "simulate-batch";
  if (verb == "--help" || verb == "-h" || argc == 1) {
    std::printf("usage: simulate --config CONFIG --out OUT [--seed N]\n"
                "       simulate-batch --batch-dir DIR --out-dir DIR\n");
    return argc == 1 ? 2 : 0;
  }
  return batch ? cmd_simulate_batch(argc, argv) : cmd_simulate(argc, argv);
}
