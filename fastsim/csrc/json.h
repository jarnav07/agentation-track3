// Minimal JSON DOM, enough to read a scenario exactly as Python's json module would:
// objects keep insertion order with "last duplicate wins" lookups, integers stay integers
// (anything that does not fit int64 is flagged so the caller can defer to the Python path),
// floats are parsed with strtod (correctly rounded, like Python's float()).
#pragma once
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fastsim {
namespace json {

enum Type { NUL, BOOL, INT, FLOAT, STRING, ARRAY, OBJECT };

struct Value {
  Type type = NUL;
  bool b = false;
  int64_t i = 0;
  bool big_int = false;  // integer literal outside int64
  double d = 0.0;
  std::string s;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  const Value* get(const char* key) const {
    if (type != OBJECT) return nullptr;
    const Value* found = nullptr;
    for (auto& kv : obj)
      if (kv.first == key) found = &kv.second;  // last duplicate wins (dict semantics)
    return found;
  }
  bool has(const char* key) const { return get(key) != nullptr; }
  // Python truthiness
  bool truthy() const {
    switch (type) {
      case NUL: return false;
      case BOOL: return b;
      case INT: return big_int || i != 0;
      case FLOAT: return d != 0.0;
      case STRING: return !s.empty();
      case ARRAY: return !arr.empty();
      case OBJECT: return !obj.empty();
    }
    return false;
  }
};

class Parser {
 public:
  Parser(const char* p, const char* end) : p_(p), end_(end) {}
  bool parse(Value& out) {
    ws();
    if (!value(out, 0)) return false;
    ws();
    return p_ == end_;
  }

 private:
  const char* p_;
  const char* end_;

  void ws() {
    while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) p_++;
  }
  bool lit(const char* s) {
    size_t n = std::strlen(s);
    if ((size_t)(end_ - p_) < n || std::memcmp(p_, s, n) != 0) return false;
    p_ += n;
    return true;
  }
  bool value(Value& v, int depth) {
    if (depth > 200 || p_ >= end_) return false;
    char c = *p_;
    if (c == '{') return object(v, depth);
    if (c == '[') return array(v, depth);
    if (c == '"') { v.type = STRING; return string(v.s); }
    if (c == 't') { v.type = BOOL; v.b = true; return lit("true"); }
    if (c == 'f') { v.type = BOOL; v.b = false; return lit("false"); }
    if (c == 'n') { v.type = NUL; return lit("null"); }
    if (c == 'N') { v.type = FLOAT; v.d = NAN; return lit("NaN"); }
    if (c == 'I') { v.type = FLOAT; v.d = INFINITY; return lit("Infinity"); }
    if (c == '-' && p_ + 1 < end_ && p_[1] == 'I') { v.type = FLOAT; v.d = -INFINITY; return lit("-Infinity"); }
    return number(v);
  }
  bool object(Value& v, int depth) {
    v.type = OBJECT;
    p_++;
    ws();
    if (p_ < end_ && *p_ == '}') { p_++; return true; }
    while (true) {
      ws();
      if (p_ >= end_ || *p_ != '"') return false;
      std::string key;
      if (!string(key)) return false;
      ws();
      if (p_ >= end_ || *p_ != ':') return false;
      p_++;
      ws();
      v.obj.emplace_back(std::move(key), Value{});
      if (!value(v.obj.back().second, depth + 1)) return false;
      ws();
      if (p_ < end_ && *p_ == ',') { p_++; continue; }
      if (p_ < end_ && *p_ == '}') { p_++; return true; }
      return false;
    }
  }
  bool array(Value& v, int depth) {
    v.type = ARRAY;
    p_++;
    ws();
    if (p_ < end_ && *p_ == ']') { p_++; return true; }
    while (true) {
      ws();
      v.arr.emplace_back();
      if (!value(v.arr.back(), depth + 1)) return false;
      ws();
      if (p_ < end_ && *p_ == ',') { p_++; continue; }
      if (p_ < end_ && *p_ == ']') { p_++; return true; }
      return false;
    }
  }
  static void utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
  }
  bool hex4(uint32_t& out) {
    if (end_ - p_ < 4) return false;
    out = 0;
    for (int k = 0; k < 4; k++) {
      char c = *p_++;
      out <<= 4;
      if (c >= '0' && c <= '9') out |= (uint32_t)(c - '0');
      else if (c >= 'a' && c <= 'f') out |= (uint32_t)(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') out |= (uint32_t)(c - 'A' + 10);
      else return false;
    }
    return true;
  }
  bool string(std::string& s) {
    p_++;  // opening quote
    while (p_ < end_) {
      char c = *p_++;
      if (c == '"') return true;
      if ((unsigned char)c < 0x20) return false;
      if (c != '\\') { s += c; continue; }
      if (p_ >= end_) return false;
      char e = *p_++;
      switch (e) {
        case '"': s += '"'; break;
        case '\\': s += '\\'; break;
        case '/': s += '/'; break;
        case 'b': s += '\b'; break;
        case 'f': s += '\f'; break;
        case 'n': s += '\n'; break;
        case 'r': s += '\r'; break;
        case 't': s += '\t'; break;
        case 'u': {
          uint32_t cp;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
            const char* save = p_;
            p_ += 2;
            uint32_t lo;
            if (hex4(lo) && lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else p_ = save;
          }
          utf8(s, cp);
          break;
        }
        default: return false;
      }
    }
    return false;
  }
  bool number(Value& v) {
    const char* start = p_;
    if (p_ < end_ && *p_ == '-') p_++;
    if (p_ >= end_) return false;
    if (*p_ == '0') p_++;
    else if (*p_ >= '1' && *p_ <= '9') while (p_ < end_ && *p_ >= '0' && *p_ <= '9') p_++;
    else return false;
    bool is_float = false;
    if (p_ < end_ && *p_ == '.') {
      is_float = true;
      p_++;
      if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
      while (p_ < end_ && *p_ >= '0' && *p_ <= '9') p_++;
    }
    if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
      is_float = true;
      p_++;
      if (p_ < end_ && (*p_ == '+' || *p_ == '-')) p_++;
      if (p_ >= end_ || *p_ < '0' || *p_ > '9') return false;
      while (p_ < end_ && *p_ >= '0' && *p_ <= '9') p_++;
    }
    std::string tok(start, p_);
    if (is_float) {
      v.type = FLOAT;
      v.d = std::strtod(tok.c_str(), nullptr);
    } else {
      v.type = INT;
      errno = 0;
      long long x = std::strtoll(tok.c_str(), nullptr, 10);
      if (errno == ERANGE) v.big_int = true;
      v.i = x;
    }
    return true;
  }
};

inline bool parse(const std::string& text, Value& out) {
  Parser p(text.data(), text.data() + text.size());
  return p.parse(out);
}

}  // namespace json
}  // namespace fastsim
