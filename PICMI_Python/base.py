"""base code for the PICMI standard
"""
import functools
import inspect
import threading
from itertools import repeat
import warnings
from typing import Self
from pydantic import model_serializer, model_validator, BaseModel, SerializeAsAny, ConfigDict


# Tracks which (instance, validator) pairs are currently executing, per thread, so that a
# mode="after" validator that assigns to ``self`` does not re-enter itself when
# ``validate_assignment=True`` re-validates each of those assignments.
_validators_in_progress = threading.local()


def resolve_once(validator):
    """Make a ``mode="after"`` model validator safe under ``validate_assignment=True``.

    With assignment validation enabled, every ``self.x = ...`` performed inside an
    after-validator triggers a re-validation, which re-runs the very same validator and
    would recurse without bound. This wrapper turns a re-entrant call *on the same
    instance* into a no-op (it returns ``self`` unchanged), so the validator's own
    derived-field assignments do not re-execute its body. The outermost call still runs in
    full, so derived fields are computed/resolved exactly once per validation.

    Apply it *under* ``@model_validator(mode="after")``::

        @model_validator(mode="after")
        @resolve_once
        def _resolve(self) -> Self:
            ...
    """
    @functools.wraps(validator)
    def wrapper(self):
        active = _validators_in_progress.__dict__.setdefault("markers", set())
        marker = (id(self), validator)
        if marker in active:
            return self
        active.add(marker)
        try:
            return validator(self)
        finally:
            active.discard(marker)
    return wrapper

codename = None

# --- The list of supported codes is needed to allow checking for bad arguments.
supported_codes = ['warp', 'warpx', 'fbpic']

def register_codename(_codename):
    """This must be called by the implementing code, passing in the code name"""
    global codename
    codename = _codename

# --- This needs to be set by the implementing package (by calling register_constants).
# --- It allows constants to be used within the picmi interface, with the constants
# --- defined in the implementation.
_implementation_constants = None

def register_constants(implementation_constants):
    """This must be called by the implementing code, passing in the constans object

    Parameters
    ----------
    implementation_constants: python object
        The object must have as attributes the physical constants
    """
    global _implementation_constants
    _implementation_constants = implementation_constants

def _get_constants():
    return _implementation_constants


class _DocumentedMetaClass(type):
    """This is used as a metaclass that combines the __doc__ of the picmistandard base and of the implementation"""
    def __new__(cls, name, bases, attrs):
        # "if bases" skips this for the _ClassWithInit (which has no bases)
        # "if bases[0].__doc__ is not None" skips this for the picmistandard classes since their bases[0] (i.e. _ClassWithInit)
        # has no __doc__.
        if bases and bases[0].__doc__ is not None:
            implementation_doc = attrs.get('__doc__', '')
            if implementation_doc:
                # The double return "\n\n" separates the picmistandard docstring from the
                # implementation-specific one, starting a new paragraph in the documentation.
                attrs['__doc__'] = bases[0].__doc__ + "\n\n" + implementation_doc
            else:
                attrs['__doc__'] = bases[0].__doc__
        return super(_DocumentedMetaClass, cls).__new__(cls, name, bases, attrs)


class _DocumentedModelMetaClass(type(BaseModel)):
    """Pydantic-compatible variant of _DocumentedMetaClass.

    It combines the __doc__ of the picmistandard base and of the implementation, so that
    downstream codes (e.g. WarpX) can extend the documentation of a pydantic PICMI class
    simply by adding a docstring to their subclass. It derives from pydantic's metaclass
    (``type(BaseModel)`` is ``ModelMetaclass``) so that it composes with ``BaseModel``.
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Skip the infrastructure base itself (its only base is BaseModel) and any class
        # whose first base carries no docstring (e.g. _PICMIModel), mirroring the guard in
        # _DocumentedMetaClass.
        if bases and bases[0] is not BaseModel and bases[0].__doc__ is not None:
            implementation_doc = namespace.get('__doc__', '')
            if implementation_doc:
                # See _DocumentedMetaClass for the rationale of this exact format.
                namespace['__doc__'] = bases[0].__doc__ + "\n\n" + implementation_doc
            else:
                namespace['__doc__'] = bases[0].__doc__
        return super().__new__(mcs, name, bases, namespace, **kwargs)


# --- Serialized PICMI objects carry the class they were dumped from under this key, so
# --- that loading them restores the same (e.g. code-specific) class, also when nested in
# --- another PICMI object or in a field that is not typed with a specific class.
PICMI_CLASS_KEY = "picmi_class"

# --- All pydantic-based PICMI classes (of the standard and of the implementing codes),
# --- by their _picmi_class_name. Only these classes are instantiated when loading data.
_picmi_classes = {}


def _picmi_class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _registered_picmi_class(name):
    try:
        return _picmi_classes[name]
    except KeyError:
        raise ValueError(
            f"Unknown {PICMI_CLASS_KEY} '{name}'. Import the module that defines this class before loading the data."
        ) from None


def _instantiate_picmi_objects(value):
    """Turn (possibly nested in lists or tuples) dictionaries written by ``model_dump`` into
    instances of the PICMI class recorded in them."""
    if isinstance(value, dict) and PICMI_CLASS_KEY in value:
        return _registered_picmi_class(value[PICMI_CLASS_KEY]).model_validate(value)
    if isinstance(value, list):
        return [_instantiate_picmi_objects(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_instantiate_picmi_objects(item) for item in value)
    return value


class _PICMIModel(BaseModel, metaclass=_DocumentedModelMetaClass):
    # Shared configuration for all pydantic-based PICMI classes.
    # - ``extra="forbid"`` restores the old behaviour of raising on unexpected keyword
    #   arguments (pydantic's default silently ignores them).
    # - ``populate_by_name`` lets downstream codes expose extension inputs under a
    #   ``<code>_`` alias while keeping their internal attribute name.
    # - ``arbitrary_types_allowed`` is needed while some referenced objects (grids,
    #   solvers, code-specific helper objects) are not yet pydantic models.
    # - ``polymorphic_serialization`` serializes an object with the fields of its actual
    #   class, e.g., a downstream grid passed to a solver keeps its code-specific fields
    #   (by default, pydantic uses the fields of the annotated standard class).
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
        polymorphic_serialization=True,
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        _picmi_classes[_picmi_class_name(cls)] = cls

    def __setattr__(self, name, value):
        # Pydantic applies an assignment before running the model validators, and keeps it
        # if they reject it. Restore the previous state in that case, so that a failed
        # assignment (including the assignments that validators make to derived fields)
        # leaves the object unchanged and valid.
        if name not in type(self).model_fields:
            return super().__setattr__(name, value)
        previous_fields = dict(self.__dict__)
        previous_fields_set = set(self.__pydantic_fields_set__)
        previous_private = None if self.__pydantic_private__ is None else dict(self.__pydantic_private__)
        try:
            super().__setattr__(name, value)
        except Exception:
            object.__setattr__(self, "__dict__", previous_fields)
            object.__setattr__(self, "__pydantic_fields_set__", previous_fields_set)
            object.__setattr__(self, "__pydantic_private__", previous_private)
            raise

    @model_serializer(mode="wrap")
    def _serialize_with_picmi_class(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            data = {PICMI_CLASS_KEY: _picmi_class_name(type(self)), **data}
        return data

    @model_validator(mode="before")
    @classmethod
    def _load_picmi_classes(cls, data):
        # Counterpart of _serialize_with_picmi_class: nested serialized PICMI objects are
        # loaded as the class they were dumped from, instead of the (standard) class a field
        # is annotated with, or a plain dictionary for fields that are not typed.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if PICMI_CLASS_KEY in data:
            dumped_class = _registered_picmi_class(data.pop(PICMI_CLASS_KEY))
            if not issubclass(cls, dumped_class):
                raise ValueError(
                    f"The data was dumped from {_picmi_class_name(dumped_class)} and cannot be loaded as {_picmi_class_name(cls)}."
                )
        return {key: _instantiate_picmi_objects(value) for key, value in data.items()}

    # PICMI objects are mutable handles to distinct entities of a simulation: two species
    # with identical parameters are still two species. Keep the identity-based equality and
    # hashing of the pre-pydantic classes (pydantic's default compares by value, which makes
    # mutable models unhashable), so that instances can be used as dictionary keys, e.g., to
    # give per-species diagnostic options. Compare ``model_dump()`` results to compare values.
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    @model_validator(mode="before")
    @classmethod
    def _ignore_other_codes_arguments(cls, data):
        # Mirror the old handle_init() behaviour: keyword arguments prefixed with the name
        # of *another* supported code are silently ignored, so that a single PICMI input
        # script can carry code-specific arguments for several codes at once. Arguments
        # prefixed with the active codename (or otherwise unknown arguments) are left in
        # place and validated normally, so that genuine typos are still reported thanks to
        # ``extra="forbid"``.
        if not isinstance(data, dict):
            return data
        return {
            k: v for k, v in data.items()
            if not ((prefix := k.split('_')[0]) in supported_codes and prefix != codename)
        }


class _ClassWithInit(metaclass=_DocumentedMetaClass):
    def handle_init(self, kw):
        # --- Grab all keywords for the current code.
        # --- Arguments for other supported codes are ignored.
        # --- If there is anything left over, it is an error.
        codekw = {}
        for k,v in kw.copy().items():
            code = k.split('_')[0]
            if code == codename:
                codekw[k] = v
                kw.pop(k)
            elif code in supported_codes:
                kw.pop(k)

        if kw:
            raise TypeError('Unexpected keyword argument: "%s"'%list(kw))

        # --- It is expected that init strips accepted keywords from codekw.
        self.init(codekw)

        if codekw:
            raise TypeError("Unexpected keyword argument for %s: '%s'"%(codename, list(codekw)))

    def init(self, kw):
        # --- The implementation of this routine should use kw.pop() to retrieve input arguments from kw.
        # --- This allows testing for any unused arguments and raising an error if found.
        pass

    def _check_unsupported_argument(self, arg_name, message=None, raise_error=False):
        """Raise a warning or exception if an unsupported argument was specified by the user

        Parameters
        ----------
        arg_name: string
            The name of the unsupported argument

        message: string
            Information to include in the warning/error message

        raise_error: bool
            If False (the default), raise a warning. If true, raise an exception
            (which interrupts the code).

        Implementation note: This should be called in the "init" method of the
        implementing class for each unsupported argument. For example, for the
        'density_scale' argument of Species:

            self._check_unsupported_argument(
                'density_scale',
                message='My code can not handle a density_scale')

        """

        # This compares the value of the parameter with the dault value in the __init__ method.
        # If they differ, this means that the user supplied a value, so a warning or error is raised.
        signature = inspect.signature(self.__init__)
        default_value = signature.parameters[arg_name].default
        if not (getattr(self, arg_name) == default_value):
            full_message = f'{self.__name__}: For argument {arg_name} is not supported.'
            if message is not None:
                full_message += f' {message}'
            if raise_error:
                raise Exception(full_message)
            else:
                warnings.warn(full_message)

    def _unsupported_value(self, arg_name, message='', raise_error=True):
        """Raise a warning or exception for argument with an unsupported value.

        Parameters
        ----------
        arg_name: string
            The name of the argument with an unsupported value

        message: string
            Information to include in the warning/error message

        raise_error: bool
            If False (the default), raise a warning. If true, raise an exception
            (which interrupts the code).

        Implementation note: This should be called when the implementing code handles
        the input arguments. For example, for 'method' in Species:

            if self.method not in ['Boris', 'Li']:
                self._unsupported_value(
                    'method',
                    message='My code only supports Boris and Li')

        """
        full_message = f'{self.__name__}: For argument {arg_name}, the value {getattr(self, arg_name)} is not supported.'
        if message is not None:
            full_message += f' {message}'
        if raise_error:
            raise Exception(full_message)
        else:
            warnings.warn(full_message)

    def _check_deprecated_argument(self, arg_name, message=None, raise_error=False):
        """Raise a warning or exception if a deprecated argument was specified by the user

        Parameters
        ----------
        arg_name: string
            The name of the deprecated argument

        message: string
            Information to include in the warning/error message

        raise_error: bool
            If False (the default), raise a warning. If true, raise an exception
            (which interrupts the code).

        Implementation note: This should be called within PICMI in the "__init__" method of classes
        for each deprecated argument. This assumes that the argument is still included in the
        argument list of __init__ as a transition until it is removed.
        For example, if the 'density_scale' argument of Species was to be deprecated:

            self._check_deprecated_argument(
                'density_scale',
                message='This argument is no longer needed')

        """

        # This compares the value of the parameter with the dault value in the __init__ method.
        # If they differ, this means that the user supplied a value, so a warning or error is raised.
        signature = inspect.signature(self.__init__)
        default_value = signature.parameters[arg_name].default
        if not (getattr(self, arg_name) == default_value):
            full_message = f'{self.__name__}: For argument {arg_name} is not supported.'
            if message is not None:
                full_message += f' {message}'
            if raise_error:
                raise Exception(full_message)
            else:
                warnings.warn(full_message)

def broadcast_validation(values, condition, message="Condition not met."):
    if not all(condition(value) for value in values):
        raise ValueError(f"{message} You gave: {values}.")
    return values


def with_mutually_exclusive(*args, defaults=None):
    def decorator(cls):
        def _mutually_exclusive(self) -> Self:
            # make sure we don't override previously implemented behaviour:
            parent_check = getattr(super(decorated, self), "_mutually_exclusive", None)
            if parent_check is not None:
                parent_check()
            if len(non_default := {arg: value for arg, default in zip(args, repeat(None) if defaults is None else defaults) if (value:=getattr(self, arg)) != default}) > 1:
                raise ValueError(f"The arguments {args} are mutually exclusive. You gave: {non_default=}.")
            return self

        # Create the subclass under the name of the decorated class. A class statement would
        # name it after its local variable, which then leaks into the validation error titles,
        # the JSON schema and the documentation. The docstring is inherited through
        # _DocumentedModelMetaClass.
        decorated = type(cls)(
            cls.__name__,
            (cls,),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "_mutually_exclusive": model_validator(mode="after")(_mutually_exclusive),
            },
        )
        return decorated
    return decorator

class _PICMI_Extension(BaseModel):
    pass

PICMI_Extension = SerializeAsAny[_PICMI_Extension]
