# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_nrf_Communication')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_nrf_Communication')
    _nrf_Communication = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_nrf_Communication', [dirname(__file__)])
        except ImportError:
            import _nrf_Communication
            return _nrf_Communication
        try:
            _mod = imp.load_module('_nrf_Communication', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _nrf_Communication = swig_import_helper()
    del swig_import_helper
else:
    import _nrf_Communication
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class nrf_Communication(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, nrf_Communication, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, nrf_Communication, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _nrf_Communication.new_nrf_Communication()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _nrf_Communication.delete_nrf_Communication
    __del__ = lambda self: None

    def setup(self, *args):
        return _nrf_Communication.nrf_Communication_setup(self, *args)

    def setSpeed(self, id, motorSpeed1, motorSpeed2, flags):
        return _nrf_Communication.nrf_Communication_setSpeed(self, id, motorSpeed1, motorSpeed2, flags)

    def setPosition(self, id, cur_pos_x, cur_pos_y, cur_angle, obj_pos_x, obj_pos_y, obj_angle, flags):
        return _nrf_Communication.nrf_Communication_setPosition(self, id, cur_pos_x, cur_pos_y, cur_angle, obj_pos_x, obj_pos_y, obj_angle, flags)

    def setConfigurationPID(self, id, kp, ki, kd, alfa, flags):
        return _nrf_Communication.nrf_Communication_setConfigurationPID(self, id, kp, ki, kd, alfa, flags)

    def sendSpeed(self, id, motorSpeed1, motorSpeed2, flags):
        return _nrf_Communication.nrf_Communication_sendSpeed(self, id, motorSpeed1, motorSpeed2, flags)

    def sendPosition(self, id, cur_pos_x, cur_pos_y, cur_angle, obj_pos_x, obj_pos_y, obj_angle, flags):
        return _nrf_Communication.nrf_Communication_sendPosition(self, id, cur_pos_x, cur_pos_y, cur_angle, obj_pos_x, obj_pos_y, obj_angle, flags)

    def sendConfigurationPID(self, id, kp, ki, kd, alfa, flags):
        return _nrf_Communication.nrf_Communication_sendConfigurationPID(self, id, kp, ki, kd, alfa, flags)

    def recv(self):
        return _nrf_Communication.nrf_Communication_recv(self)

    def getInfoRet(self, id):
        return _nrf_Communication.nrf_Communication_getInfoRet(self, id)
nrf_Communication_swigregister = _nrf_Communication.nrf_Communication_swigregister
nrf_Communication_swigregister(nrf_Communication)
DEFAULT_DEVICE_NAME = _nrf_Communication.DEFAULT_DEVICE_NAME
NRF_BUFFER_SIZE = _nrf_Communication.NRF_BUFFER_SIZE
_MSG_BEGIN = _nrf_Communication._MSG_BEGIN
_MSG_END = _nrf_Communication._MSG_END
MSG_RETURN_POSITION = _nrf_Communication.MSG_RETURN_POSITION
MSG_RETURN_BATTERY = _nrf_Communication.MSG_RETURN_BATTERY
LENGHT_SPEED = _nrf_Communication.LENGHT_SPEED
LENGHT_PID = _nrf_Communication.LENGHT_PID
LENGHT_POSITIONS = _nrf_Communication.LENGHT_POSITIONS

# This file is compatible with both classic and new-style classes.


