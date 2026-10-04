// CPython extension `fastsim._engine`: run(spec: bytes) -> dict[str, bytes] of column buffers.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "engine.h"

template <typename T>
static int put(PyObject* d, const char* key, const std::vector<T>& v) {
  PyObject* b = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(v.data()), (Py_ssize_t)(v.size() * sizeof(T)));
  if (!b) return -1;
  int rc = PyDict_SetItemString(d, key, b);
  Py_DECREF(b);
  return rc;
}

static PyObject* run(PyObject*, PyObject* args) {
  Py_buffer buf;
  if (!PyArg_ParseTuple(args, "y*", &buf)) return nullptr;
  fastsim::Spec spec;
  fastsim::Output out;
  std::string err;
  bool ok;
  Py_BEGIN_ALLOW_THREADS
  ok = fastsim::parse_spec(static_cast<const char*>(buf.buf), (size_t)buf.len, spec, err);
  if (ok) {
    ok = fastsim::run_engine(spec, out);
    if (!ok) err = out.error;
  }
  Py_END_ALLOW_THREADS
  PyBuffer_Release(&buf);
  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError, err.c_str());
    return nullptr;
  }
  PyObject* d = PyDict_New();
  if (!d) return nullptr;
  if (put(d, "t_ns", out.t_ns) || put(d, "agent_id", out.agent_id) || put(d, "msg_code", out.msg_code) ||
      put(d, "side_code", out.side_code) || put(d, "price", out.price) || put(d, "size", out.size) ||
      put(d, "order_id", out.order_id) || put(d, "l_t_recv", out.l_t_recv) || put(d, "l_t_send", out.l_t_send) ||
      put(d, "l_t_send_valid", out.l_t_send_valid) || put(d, "l_latency", out.l_latency) ||
      put(d, "l_src", out.l_src) || put(d, "l_dst", out.l_dst) || put(d, "l_msg_id", out.l_msg_id) ||
      put(d, "l_kind", out.l_kind) || put(d, "l_order_id", out.l_order_id) ||
      put(d, "l_order_valid", out.l_order_valid) || put(d, "l_causal", out.l_causal) ||
      put(d, "l_causal_valid", out.l_causal_valid)) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}

static PyMethodDef methods[] = {{"run", run, METH_VARARGS, "Run one scenario from a packed spec."},
                                {nullptr, nullptr, 0, nullptr}};
static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_engine", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__engine(void) { return PyModule_Create(&moddef); }
