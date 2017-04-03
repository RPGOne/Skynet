#include <Python.h>
#include <structmember.h>

#include <stdbool.h>
#include <TH/TH.h>
#include "THP.h"

PyObject *THPGeneratorClass = NULL;

PyObject * THPGenerator_New()
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create a new generator object - "
        "failed to allocate argument tuple");
    return NULL;
  }
  PyObject *result = PyObject_Call((PyObject*)THPGeneratorClass, args, NULL);
  Py_DECREF(args);
  return result;
}

static void THPGenerator_dealloc(THPGenerator* self)
{
  THGenerator_free(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  if ((args && PyTuple_Size(args) != 0) || kwargs) {
    THPUtils_setError("torch.Generator constructor doesn't accept any arguments");
    return NULL;
  }
  THPGeneratorPtr self = (THPGenerator *)type->tp_alloc(type, 0);
  self->cdata = THGenerator_new();

  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  THGenerator *generator = self->cdata;
  THPByteTensorPtr res = (THPByteTensor *)THPByteTensor_NewEmpty();
  if (!res) return NULL;
  THByteTensor_getRNGState(generator, res->cdata);
  return (PyObject *)res.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_setState(THPGenerator *self, PyObject *_new_state)
{
  HANDLE_TH_ERRORS
  THGenerator *generator = self->cdata;
  THPUtils_assert(THPByteTensor_Check(_new_state), "set_state expects a "
          "torch.ByteTensor, but got %s", THPUtils_typename(_new_state));
  THByteTensor *new_state = ((THPByteTensor*)_new_state)->cdata;
  THByteTensor_setRNGState(generator, new_state);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_manualSeed(THPGenerator *self, PyObject *seed)
{
  HANDLE_TH_ERRORS
  THGenerator *generator = self->cdata;
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  THRandom_manualSeed(generator, THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLong(THRandom_seed(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLong(THRandom_initialSeed(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPGenerator_methods[] = {
  {"get_state",       (PyCFunction)THPGenerator_getState,       METH_NOARGS,  NULL},
  {"set_state",       (PyCFunction)THPGenerator_setState,       METH_O,       NULL},
  {"manual_seed",     (PyCFunction)THPGenerator_manualSeed,     METH_O,       NULL},
  {"seed",            (PyCFunction)THPGenerator_seed,           METH_NOARGS,  NULL},
  {"initial_seed",    (PyCFunction)THPGenerator_initialSeed,    METH_NOARGS,  NULL},
  {NULL}
};

static struct PyMemberDef THPGenerator_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, NULL},
  {NULL}
};

PyTypeObject THPGeneratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C.Generator",                  /* tp_name */
  sizeof(THPGenerator),                  /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPGenerator_dealloc,      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPGenerator_methods,                  /* tp_methods */
  THPGenerator_members,                  /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPGenerator_pynew,                    /* tp_new */
};

bool THPGenerator_init(PyObject *module)
{
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject *)&THPGeneratorType);
  return true;
}
