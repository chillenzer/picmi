Code-specific extensions
========================

Implementing codes can provide classes that have no counterpart in the standard, e.g., additional field solvers or diagnostics.
They derive these classes from the extension class of the respective kind, so that the objects are accepted by the PICMI classes where objects of that kind are expected.
Classes that are only used by other code-specific classes derive from ``PICMI_Extension`` directly.

.. autopydantic_model:: picmistandard.PICMI_Extension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_SolverExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_DistributionExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_LayoutExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_LaserExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_LaserInjectionExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_AppliedFieldExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_DiagnosticExtension
    :inherited-members: BaseModel

.. autopydantic_model:: picmistandard.PICMI_InteractionExtension
    :inherited-members: BaseModel

Classes with analytic expressions, whose parameters are given as additional keyword arguments, derive from ``PICMI_ExpressionParameters``, too.

.. autopydantic_model:: picmistandard.PICMI_ExpressionParameters
    :inherited-members: BaseModel
