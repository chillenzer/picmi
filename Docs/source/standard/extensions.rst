Code-specific extensions
========================

Implementing codes can provide classes that have no counterpart in the standard, e.g., additional field solvers or diagnostics.
They derive these classes from the extension class of the respective kind, so that the objects are accepted by the PICMI classes where objects of that kind are expected.
Classes that are only used by other code-specific classes derive from ``PICMI_Extension`` directly.

.. autoclass:: picmistandard.PICMI_Extension

.. autoclass:: picmistandard.PICMI_SolverExtension

.. autoclass:: picmistandard.PICMI_DistributionExtension

.. autoclass:: picmistandard.PICMI_LayoutExtension

.. autoclass:: picmistandard.PICMI_LaserExtension

.. autoclass:: picmistandard.PICMI_LaserInjectionExtension

.. autoclass:: picmistandard.PICMI_AppliedFieldExtension

.. autoclass:: picmistandard.PICMI_DiagnosticExtension

.. autoclass:: picmistandard.PICMI_InteractionExtension

Classes with analytic expressions, whose parameters are given as additional keyword arguments, derive from ``PICMI_ExpressionParameters``, too.

.. autoclass:: picmistandard.PICMI_ExpressionParameters
