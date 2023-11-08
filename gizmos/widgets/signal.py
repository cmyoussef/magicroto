"""
Signal/Slot implementation
Adapted from: http://code.activestate.com/recipes/577980/ (MIT license)
"""

import contextlib
import inspect
import traceback
import weakref

from magicroto.utils.logger import logger


class Signal(object):
    """
    Callable signal with slot connection (function or methods)
    """

    def __init__(self, name="", description=""):
        self._functions = set()  # weakref.WeakSet()
        self._methods = weakref.WeakKeyDictionary()

        self.name = name
        self.description = description

        self.disabled = False

    def __call__(self, *args, **kwargs):
        if self.disabled:
            return

        # Call handler functions
        for func in self._functions:
            # func(*args, **kwargs)

            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())

        # Call handler methods
        for obj, funcs in list(self._methods.items()):
            for func in funcs:
                # func(obj, *args, **kwargs)

                try:
                    func(obj, *args, **kwargs)
                except Exception as e:
                    logger.error(e)
                    logger.error(traceback.format_exc())

    def connect(self, slot):
        """
        Connect a function or method to be called when signal is emitted
        :param slot: function or method
        """
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)

    def disconnect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)

    def clear(self):
        self._functions.clear()
        self._methods.clear()

    def emit(self, *args, **kwargs):
        """ simple wrapper to allow parity with qt signals """
        self(*args, **kwargs)

    def __str__(self):
        return "<{}> signal ({})".format(self.name, self.description)

    @contextlib.contextmanager
    def after(self):
        """
        this contextmanager disables the signal for the code block, and runs it right after

        >>> def doStuff():
        >>>     ....
        >>>     mySignal()
        >>>
        >>> with mySignal.after():
        >>>     doStuff() # this won't trigger the signal
        # mySignal gets called here instead

        """
        prevState, self.disabled = self.disabled, True
        yield
        self.disabled = prevState
        self()


# Sample usage:
if __name__ == '__main__':
    class Model(object):
        def __init__(self, value):
            self.__value = value
            self.changed = Signal()

        def setValue(self, value):
            self.__value = value
            self.changed()  # Emit signal

        def getValue(self):
            return self.__value


    class View(object):
        def __init__(self, name, model):
            self.name = name
            self.model = model
            model.changed.connect(self.model_changed)

        def model_changed(self):
            print("[{}] value: {}".format(self.name, self.model.getValue()))


    print("Beginning Tests:")
    model = Model(10)
    view1 = View('v1', model)
    view2 = View('v2', model)
    view3 = View('v3', model)

    print("Setting value to 20...")
    model.setValue(20)

    print("Deleting a view, and setting value to 30...")
    del view1
    model.setValue(30)

    print("Clearing all listeners, and setting value to 40...")
    model.changed.clear()
    model.setValue(40)

    print("Testing non-member function...")


    def bar():
        print("[bar] Calling Function!")


    model.changed.connect(bar)
    model.setValue(50)
