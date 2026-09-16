"""Classes following the PICMI standard
These should be the base classes for Python implementation of the PICMI standard
"""

from typing import ClassVar, Self, Sequence, get_args, Literal
from pydantic import BaseModel, Field, model_validator

from .base import _ClassWithInit, _PICMIModel, resolve_once


def _fill_per_axis(per_axis_values, fallback):
    """Combine optional per-axis values with a fallback vector of the same length.

    Each axis that was not given (``None``) takes the value of ``fallback`` on that axis.
    This is used for the particle boundaries of the grids, which default to the field
    boundaries, including when only some of the per-axis particle values are specified.
    """
    return [
        fallback_value if value is None else value
        for value, fallback_value in zip(per_axis_values, fallback)
    ]


class PICMI_BinomialSmoother(_PICMIModel):
    """
    Describes a binomial smoother operator (applied to grids).
    """

    n_pass: Sequence[int] | None = Field(
        default=None,
        description="Vector of integers. Number of passes along each axis",
    )
    compensation: Sequence[bool] | None = Field(
        default=None, description="Flags whether to apply compensation along each axis"
    )
    stride: Sequence[int] | None = Field(
        default=None, description="Stride along each axis"
    )
    alpha: Sequence[float] | None = Field(
        default=None, description="Smoothing coefficients along each axis"
    )


class PICMI_Cartesian1DGrid(_PICMIModel):
    """
    One-dimensional Cartesian grid
    Parameters can be specified either as vectors or separately.
    (If both are specified, the vector is used.)

    References
    ----------
    absorbing_silver_mueller: A local absorbing boundary condition that works best under normal incidence angle.
    Based on the Silver-Mueller Radiation Condition, e.g., in

    * A. K. Belhora and L. Pichon, "Maybe Efficient Absorbing Boundary Conditions for the Finite Element Solution of 3D Scattering Problems," 1995,
      https://doi.org/10.1109/20.376322
    * B Engquist and A. Majdat, "Absorbing boundary conditions for numerical simulation of waves," 1977,
      https://doi.org/10.1073/pnas.74.5.1765
    * R. Lehe, "Electromagnetic wave propagation in Particle-In-Cell codes," 2016,
      US Particle Accelerator School (USPAS) Summer Session, Self-Consistent Simulations of Beam and Plasma Systems
      https://people.nscl.msu.edu/~lund/uspas/scs_2016/lec_adv/A1b_EM_Waves.pdf
    """

    # Note for implementations, as a matter of convenience and flexibility, the user interface allows
    # specifying various quantities using either the individual named attributes (such as nx) or a
    # vector of values (such as number_of_cells). However, internally, only the vectors are saved and
    # the implementation needs to use the those to access the user input.

    number_of_dimensions: ClassVar[int] = 1

    # Vector forms (the internally-used representation)
    number_of_cells: list[int] | None = Field(
        default=None,
        description="Number of cells along each axis (number of nodes is number_of_cells+1)",
    )
    lower_bound: list[float] | None = Field(
        default=None, description="Position of the node at the lower bound [m]"
    )
    upper_bound: list[float] | None = Field(
        default=None, description="Position of the node at the upper bound [m]"
    )
    lower_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    upper_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    # Per-axis scalar forms (resolved into the vector forms during validation)
    nx: int | None = Field(
        default=None, description="Number of cells along X (number of nodes=nx+1)"
    )
    xmin: float | None = Field(
        default=None, description="Position of first node along X [m]"
    )
    xmax: float | None = Field(
        default=None, description="Position of last node along X [m]"
    )
    bc_xmin: str | None = Field(
        default=None,
        description="Boundary condition at min X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_xmax: str | None = Field(
        default=None,
        description="Boundary condition at max X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    moving_window_velocity: list[float] | None = Field(
        default=None, description="Moving frame velocity [m/s]"
    )
    refined_regions: list = Field(
        default_factory=list,
        description="List of refined regions, each element being a list of the format [level, lo, hi, refinement_factor], with level being the refinement level, with 1 being the first level of refinement, 2 being the second etc, lo and hi being vectors of length 2 specifying the extent of the region, and refinement_factor defaulting to [2,2] (relative to next lower level)",
    )
    lower_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle lower bound [m]"
    )
    upper_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle upper bound [m]"
    )
    xmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along X [m]"
    )
    xmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along X [m]"
    )
    lower_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    upper_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    bc_xmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min X for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_xmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max X for particles: One of periodic, absorbing, reflect, thermal",
    )
    guard_cells: list[int] | None = Field(
        default=None, description="Number of guard cells used along each direction"
    )
    pml_cells: list[int] | None = Field(
        default=None,
        description="Number of Perfectly Matched Layer (PML) cells along each direction",
    )

    @model_validator(mode="after")
    @resolve_once
    def _resolve_grid(self) -> Self:
        # Sanity check and init of input arguments related to grid parameters
        assert (self.number_of_cells is not None) or (
            self.nx is not None
        ), "Either number_of_cells or nx must be specified"
        assert (self.lower_bound is not None) or (
            self.xmin is not None
        ), "Either lower_bound or xmin must be specified"
        assert (self.upper_bound is not None) or (
            self.xmax is not None
        ), "Either upper_bound or xmax must be specified"
        assert (self.lower_boundary_conditions is not None) or (
            self.bc_xmin is not None
        ), "Either lower_boundary_conditions or bc_xmin"
        assert (self.upper_boundary_conditions is not None) or (
            self.bc_xmax is not None
        ), "Either upper_boundary_conditions or bc_xmax must be specified"

        if self.number_of_cells is None:
            self.number_of_cells = [self.nx]
        if self.lower_bound is None:
            self.lower_bound = [self.xmin]
        if self.upper_bound is None:
            self.upper_bound = [self.xmax]
        if self.lower_boundary_conditions is None:
            self.lower_boundary_conditions = [self.bc_xmin]
        if self.upper_boundary_conditions is None:
            self.upper_boundary_conditions = [self.bc_xmax]

        # Sanity check and init of input arguments related to particle boundary parameters
        # By default, if not specified, particle boundary values are the same as field boundary values
        # By default, if not specified, particle boundary conditions are the same as field boundary conditions
        if self.lower_bound_particles is None:
            self.lower_bound_particles = _fill_per_axis(
                [self.xmin_particles], self.lower_bound
            )
        if self.upper_bound_particles is None:
            self.upper_bound_particles = _fill_per_axis(
                [self.xmax_particles], self.upper_bound
            )

        if self.lower_boundary_conditions_particles is None:
            self.lower_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmin_particles], self.lower_boundary_conditions
            )
        if self.upper_boundary_conditions_particles is None:
            self.upper_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmax_particles], self.upper_boundary_conditions
            )

        # Sanity check on dimensionality of vector quantities
        assert len(self.number_of_cells) == 1, "Wrong number of cells specified"
        assert len(self.lower_bound) == 1, "Wrong number of lower bounds specified"
        assert len(self.upper_bound) == 1, "Wrong number of upper bounds specified"
        assert len(self.lower_boundary_conditions) == 1, (
            "Wrong number of lower boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions) == 1, (
            "Wrong number of upper boundary conditions specified"
        )
        assert len(self.lower_bound_particles) == 1, (
            "Wrong number of particle lower bounds specified"
        )
        assert len(self.upper_bound_particles) == 1, (
            "Wrong number of particle upper bounds specified"
        )
        assert len(self.lower_boundary_conditions_particles) == 1, (
            "Wrong number of lower particle boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions_particles) == 1, (
            "Wrong number of upper particle boundary conditions specified"
        )

        for region in self.refined_regions:
            if len(region) == 3:
                region.append([2])
            assert len(region[1]) == 1, (
                "The lo extent of the refined region must be a vector of length 1"
            )
            assert len(region[2]) == 1, (
                "The hi extent of the refined region must be a vector of length 1"
            )
            assert len(region[3]) == 1, (
                "The refinement factor of the refined region must be a vector of length 1"
            )

        return self

    def add_refined_region(self, level, lo, hi, refinement_factor=[2]):
        """Add a refined region.
        level: the refinement level, with 1 being the first level of refinement, 2 being the second etc.
        lo, hi: vectors of length 2 specifying the extent of the region
        refinement_factor: defaulting to [2,2] (relative to next lower level)
        """
        self.refined_regions.append([level, lo, hi, refinement_factor])


class PICMI_CylindricalGrid(_PICMIModel):
    """
    Axisymmetric, cylindrical grid
    Parameters can be specified either as vectors or separately.
    (If both are specified, the vector is used.)

    References
    ----------
    absorbing_silver_mueller: A local absorbing boundary condition that works best under normal incidence angle.
    Based on the Silver-Mueller Radiation Condition, e.g., in

    * A. K. Belhora and L. Pichon, "Maybe Efficient Absorbing Boundary Conditions for the Finite Element Solution of 3D Scattering Problems," 1995,
      https://doi.org/10.1109/20.376322
    * B Engquist and A. Majdat, "Absorbing boundary conditions for numerical simulation of waves," 1977,
      https://doi.org/10.1073/pnas.74.5.1765
    * R. Lehe, "Electromagnetic wave propagation in Particle-In-Cell codes," 2016,
      US Particle Accelerator School (USPAS) Summer Session, Self-Consistent Simulations of Beam and Plasma Systems
      https://people.nscl.msu.edu/~lund/uspas/scs_2016/lec_adv/A1b_EM_Waves.pdf
    """

    # Note for implementations, as a matter of convenience and flexibility, the user interface allows
    # specifying various quantities using either the individual named attributes (such as nr and nz) or a
    # vector of values (such as number_of_cells). However, internally, only the vectors are saved and
    # the implementation needs to use the those to access the user input.

    number_of_dimensions: ClassVar[int] = 2

    # Vector forms (the internally-used representation)
    number_of_cells: list[int] | None = Field(
        default=None,
        description="Number of cells along each axis (number of nodes is number_of_cells+1)",
    )
    lower_bound: list[float] | None = Field(
        default=None, description="Position of the node at the lower bound [m]"
    )
    upper_bound: list[float] | None = Field(
        default=None, description="Position of the node at the upper bound [m]"
    )
    lower_boundary_conditions: list[str | None] | None = Field(
        default=None,
        description="Conditions at lower boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    upper_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    # Per-axis scalar forms (resolved into the vector forms during validation)
    nr: int | None = Field(
        default=None, description="Number of cells along R (number of nodes=nr+1)"
    )
    nz: int | None = Field(
        default=None, description="Number of cells along Z (number of nodes=nz+1)"
    )
    n_azimuthal_modes: int | None = Field(
        default=None, description="Number of azimuthal modes"
    )
    rmin: float | None = Field(
        default=None, description="Position of first node along R [m]"
    )
    rmax: float | None = Field(
        default=None, description="Position of last node along R [m]"
    )
    zmin: float | None = Field(
        default=None, description="Position of first node along Z [m]"
    )
    zmax: float | None = Field(
        default=None, description="Position of last node along Z [m]"
    )
    bc_rmin: str | None = Field(
        default=None,
        description="Boundary condition at min R: One of open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_rmax: str | None = Field(
        default=None,
        description="Boundary condition at max R: One of open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_zmin: str | None = Field(
        default=None,
        description="Boundary condition at min Z: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_zmax: str | None = Field(
        default=None,
        description="Boundary condition at max Z: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    moving_window_velocity: list[float] | None = Field(
        default=None, description="Moving frame velocity [m/s]"
    )
    refined_regions: list = Field(
        default_factory=list,
        description="List of refined regions, each element being a list of the format [level, lo, hi, refinement_factor], with level being the refinement level, with 1 being the first level of refinement, 2 being the second etc, lo and hi being vectors of length 2 specifying the extent of the region, and refinement_factor defaulting to [2,2] (relative to next lower level)",
    )
    lower_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle lower bound [m]"
    )
    upper_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle upper bound [m]"
    )
    rmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along R [m]"
    )
    rmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along R [m]"
    )
    zmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along Z [m]"
    )
    zmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along Z [m]"
    )
    # --Like bc_rmin, the radial entry may be None since the lower radial boundary will usually be the axis.
    lower_boundary_conditions_particles: list[str | None] | None = Field(
        default=None,
        description="Conditions at lower boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    upper_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    bc_rmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min R for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_rmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max R for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_zmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min Z for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_zmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max Z for particles: One of periodic, absorbing, reflect, thermal",
    )
    guard_cells: list[int] | None = Field(
        default=None, description="Number of guard cells used along each direction"
    )
    pml_cells: list[int] | None = Field(
        default=None,
        description="Number of Perfectly Matched Layer (PML) cells along each direction",
    )

    @model_validator(mode="after")
    @resolve_once
    def _resolve_grid(self) -> Self:
        # Sanity check and init of input arguments related to grid parameters
        assert (self.number_of_cells is not None) or (
            self.nr is not None and self.nz is not None
        ), "Either number_of_cells or nr and nz must be specified"
        assert (self.lower_bound is not None) or (
            self.rmin is not None and self.zmin is not None
        ), "Either lower_bound or rmin and zmin must be specified"
        assert (self.upper_bound is not None) or (
            self.rmax is not None and self.zmax is not None
        ), "Either upper_bound or rmax and zmax must be specified"
        # --Allow bc_rmin to be None since it will usually be the axis.
        assert (self.lower_boundary_conditions is not None) or (
            self.bc_zmin is not None
        ), "Either lower_boundary_conditions or bc_rmin and bc_zmin must be specified"
        assert (self.upper_boundary_conditions is not None) or (
            self.bc_rmax is not None and self.bc_zmax is not None
        ), "Either upper_boundary_conditions or bc_rmax and bc_zmax must be specified"

        if self.number_of_cells is None:
            self.number_of_cells = [self.nr, self.nz]
        if self.lower_bound is None:
            self.lower_bound = [self.rmin, self.zmin]
        if self.upper_bound is None:
            self.upper_bound = [self.rmax, self.zmax]
        if self.lower_boundary_conditions is None:
            self.lower_boundary_conditions = [self.bc_rmin, self.bc_zmin]
        if self.upper_boundary_conditions is None:
            self.upper_boundary_conditions = [self.bc_rmax, self.bc_zmax]

        # Sanity check and init of input arguments related to particle boundary parameters
        # By default, if not specified, particle boundary values are the same as field boundary values
        # By default, if not specified, particle boundary conditions are the same as field boundary conditions
        if self.lower_bound_particles is None:
            self.lower_bound_particles = _fill_per_axis(
                [self.rmin_particles, self.zmin_particles], self.lower_bound
            )
        if self.upper_bound_particles is None:
            self.upper_bound_particles = _fill_per_axis(
                [self.rmax_particles, self.zmax_particles], self.upper_bound
            )

        if self.lower_boundary_conditions_particles is None:
            self.lower_boundary_conditions_particles = _fill_per_axis(
                [self.bc_rmin_particles, self.bc_zmin_particles],
                self.lower_boundary_conditions,
            )
        if self.upper_boundary_conditions_particles is None:
            self.upper_boundary_conditions_particles = _fill_per_axis(
                [self.bc_rmax_particles, self.bc_zmax_particles],
                self.upper_boundary_conditions,
            )

        # Sanity check on dimensionality of vector quantities
        assert len(self.number_of_cells) == 2, "Wrong number of cells specified"
        assert len(self.lower_bound) == 2, "Wrong number of lower bounds specified"
        assert len(self.upper_bound) == 2, "Wrong number of upper bounds specified"
        assert len(self.lower_boundary_conditions) == 2, (
            "Wrong number of lower boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions) == 2, (
            "Wrong number of upper boundary conditions specified"
        )

        for region in self.refined_regions:
            if len(region) == 3:
                region.append([2, 2])
            assert len(region[1]) == 2, (
                "The lo extent of the refined region must be a vector of length 2"
            )
            assert len(region[2]) == 2, (
                "The hi extent of the refined region must be a vector of length 2"
            )
            assert len(region[3]) == 2, (
                "The refinement factor of the refined region must be a vector of length 2"
            )

        return self

    def add_refined_region(self, level, lo, hi, refinement_factor=[2, 2]):
        """Add a refined region.
        level: the refinement level, with 1 being the first level of refinement, 2 being the second etc.
        lo, hi: vectors of length 2 specifying the extent of the region
        refinement_factor: defaulting to [2,2] (relative to next lower level)
        """
        self.refined_regions.append([level, lo, hi, refinement_factor])


class PICMI_Cartesian2DGrid(_PICMIModel):
    """
    Two dimensional Cartesian grid
    Parameters can be specified either as vectors or separately.
    (If both are specified, the vector is used.)

    References
    ----------
    absorbing_silver_mueller: A local absorbing boundary condition that works best under normal incidence angle.
    Based on the Silver-Mueller Radiation Condition, e.g., in

    * A. K. Belhora and L. Pichon, "Maybe Efficient Absorbing Boundary Conditions for the Finite Element Solution of 3D Scattering Problems," 1995,
      https://doi.org/10.1109/20.376322
    * B Engquist and A. Majdat, "Absorbing boundary conditions for numerical simulation of waves," 1977,
      https://doi.org/10.1073/pnas.74.5.1765
    * R. Lehe, "Electromagnetic wave propagation in Particle-In-Cell codes," 2016,
      US Particle Accelerator School (USPAS) Summer Session, Self-Consistent Simulations of Beam and Plasma Systems
      https://people.nscl.msu.edu/~lund/uspas/scs_2016/lec_adv/A1b_EM_Waves.pdf
    """

    # Note for implementations, as a matter of convenience and flexibility, the user interface allows
    # specifying various quantities using either the individual named attributes (such as nx and ny) or a
    # vector of values (such as number_of_cells). However, internally, only the vectors are saved and
    # the implementation needs to use the those to access the user input.

    number_of_dimensions: ClassVar[int] = 2

    # Vector forms (the internally-used representation)
    number_of_cells: list[int] | None = Field(
        default=None,
        description="Number of cells along each axis (number of nodes is number_of_cells+1)",
    )
    lower_bound: list[float] | None = Field(
        default=None, description="Position of the node at the lower bound [m]"
    )
    upper_bound: list[float] | None = Field(
        default=None, description="Position of the node at the upper bound [m]"
    )
    lower_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    upper_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    # Per-axis scalar forms (resolved into the vector forms during validation)
    nx: int | None = Field(
        default=None, description="Number of cells along X (number of nodes=nx+1)"
    )
    ny: int | None = Field(
        default=None, description="Number of cells along Y (number of nodes=ny+1)"
    )
    xmin: float | None = Field(
        default=None, description="Position of first node along X [m]"
    )
    xmax: float | None = Field(
        default=None, description="Position of last node along X [m]"
    )
    ymin: float | None = Field(
        default=None, description="Position of first node along Y [m]"
    )
    ymax: float | None = Field(
        default=None, description="Position of last node along Y [m]"
    )
    bc_xmin: str | None = Field(
        default=None,
        description="Boundary condition at min X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_xmax: str | None = Field(
        default=None,
        description="Boundary condition at max X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_ymin: str | None = Field(
        default=None,
        description="Boundary condition at min Y: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_ymax: str | None = Field(
        default=None,
        description="Boundary condition at max Y: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    moving_window_velocity: list[float] | None = Field(
        default=None, description="Moving frame velocity [m/s]"
    )
    refined_regions: list = Field(
        default_factory=list,
        description="List of refined regions, each element being a list of the format [level, lo, hi, refinement_factor], with level being the refinement level, with 1 being the first level of refinement, 2 being the second etc, lo and hi being vectors of length 2 specifying the extent of the region, and refinement_factor defaulting to [2,2] (relative to next lower level)",
    )
    lower_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle lower bound [m]"
    )
    upper_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle upper bound [m]"
    )
    xmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along X [m]"
    )
    xmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along X [m]"
    )
    ymin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along Y [m]"
    )
    ymax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along Y [m]"
    )
    lower_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    upper_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    bc_xmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min X for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_xmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max X for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_ymin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min Y for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_ymax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max Y for particles: One of periodic, absorbing, reflect, thermal",
    )
    guard_cells: list[int] | None = Field(
        default=None, description="Number of guard cells used along each direction"
    )
    pml_cells: list[int] | None = Field(
        default=None,
        description="Number of Perfectly Matched Layer (PML) cells along each direction",
    )

    @model_validator(mode="after")
    @resolve_once
    def _resolve_grid(self) -> Self:
        # Sanity check and init of input arguments related to grid parameters
        assert (self.number_of_cells is not None) or (
            self.nx is not None and self.ny is not None
        ), "Either number_of_cells or nx and ny must be specified"
        assert (self.lower_bound is not None) or (
            self.xmin is not None and self.ymin is not None
        ), "Either lower_bound or xmin and ymin must be specified"
        assert (self.upper_bound is not None) or (
            self.xmax is not None and self.ymax is not None
        ), "Either upper_bound or xmax and ymax must be specified"
        assert (self.lower_boundary_conditions is not None) or (
            self.bc_xmin is not None and self.bc_ymin is not None
        ), "Either lower_boundary_conditions or bc_xmin and bc_ymin must be specified"
        assert (self.upper_boundary_conditions is not None) or (
            self.bc_xmax is not None and self.bc_ymax is not None
        ), "Either upper_boundary_conditions or bc_xmax and bc_ymax must be specified"

        if self.number_of_cells is None:
            self.number_of_cells = [self.nx, self.ny]
        if self.lower_bound is None:
            self.lower_bound = [self.xmin, self.ymin]
        if self.upper_bound is None:
            self.upper_bound = [self.xmax, self.ymax]
        if self.lower_boundary_conditions is None:
            self.lower_boundary_conditions = [self.bc_xmin, self.bc_ymin]
        if self.upper_boundary_conditions is None:
            self.upper_boundary_conditions = [self.bc_xmax, self.bc_ymax]

        # Sanity check and init of input arguments related to particle boundary parameters
        # By default, if not specified, particle boundary values are the same as field boundary values
        # By default, if not specified, particle boundary conditions are the same as field boundary conditions
        if self.lower_bound_particles is None:
            self.lower_bound_particles = _fill_per_axis(
                [self.xmin_particles, self.ymin_particles], self.lower_bound
            )
        if self.upper_bound_particles is None:
            self.upper_bound_particles = _fill_per_axis(
                [self.xmax_particles, self.ymax_particles], self.upper_bound
            )

        if self.lower_boundary_conditions_particles is None:
            self.lower_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmin_particles, self.bc_ymin_particles],
                self.lower_boundary_conditions,
            )
        if self.upper_boundary_conditions_particles is None:
            self.upper_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmax_particles, self.bc_ymax_particles],
                self.upper_boundary_conditions,
            )

        # Sanity check on dimensionality of vector quantities
        assert len(self.number_of_cells) == 2, "Wrong number of cells specified"
        assert len(self.lower_bound) == 2, "Wrong number of lower bounds specified"
        assert len(self.upper_bound) == 2, "Wrong number of upper bounds specified"
        assert len(self.lower_boundary_conditions) == 2, (
            "Wrong number of lower boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions) == 2, (
            "Wrong number of upper boundary conditions specified"
        )
        assert len(self.lower_bound_particles) == 2, (
            "Wrong number of particle lower bounds specified"
        )
        assert len(self.upper_bound_particles) == 2, (
            "Wrong number of particle upper bounds specified"
        )
        assert len(self.lower_boundary_conditions_particles) == 2, (
            "Wrong number of lower particle boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions_particles) == 2, (
            "Wrong number of upper particle boundary conditions specified"
        )

        for region in self.refined_regions:
            if len(region) == 3:
                region.append([2, 2])
            assert len(region[1]) == 2, (
                "The lo extent of the refined region must be a vector of length 2"
            )
            assert len(region[2]) == 2, (
                "The hi extent of the refined region must be a vector of length 2"
            )
            assert len(region[3]) == 2, (
                "The refinement factor of the refined region must be a vector of length 2"
            )

        return self

    def add_refined_region(self, level, lo, hi, refinement_factor=[2, 2]):
        """Add a refined region.
        level: the refinement level, with 1 being the first level of refinement, 2 being the second etc.
        lo, hi: vectors of length 2 specifying the extent of the region
        refinement_factor: defaulting to [2,2] (relative to next lower level)
        """
        self.refined_regions.append([level, lo, hi, refinement_factor])


class PICMI_Cartesian3DGrid(_PICMIModel):
    """
    Three dimensional Cartesian grid
    Parameters can be specified either as vectors or separately.
    (If both are specified, the vector is used.)

    References
    ----------
    absorbing_silver_mueller: A local absorbing boundary condition that works best under normal incidence angle.
    Based on the Silver-Mueller Radiation Condition, e.g., in

    * A. K. Belhora and L. Pichon, "Maybe Efficient Absorbing Boundary Conditions for the Finite Element Solution of 3D Scattering Problems," 1995,
      https://doi.org/10.1109/20.376322
    * B Engquist and A. Majdat, "Absorbing boundary conditions for numerical simulation of waves," 1977,
      https://doi.org/10.1073/pnas.74.5.1765
    * R. Lehe, "Electromagnetic wave propagation in Particle-In-Cell codes," 2016,
      US Particle Accelerator School (USPAS) Summer Session, Self-Consistent Simulations of Beam and Plasma Systems
      https://people.nscl.msu.edu/~lund/uspas/scs_2016/lec_adv/A1b_EM_Waves.pdf
    """

    # Note for implementations, as a matter of convenience and flexibility, the user interface allows
    # specifying various quantities using either the individual named attributes (such as nx, ny, and nz) or a
    # vector of values (such as number_of_cells). However, internally, only the vectors are saved and
    # the implementation needs to use the those to access the user input.

    number_of_dimensions: ClassVar[int] = 3

    # Vector forms (the internally-used representation)
    number_of_cells: list[int] | None = Field(
        default=None,
        description="Number of cells along each axis (number of nodes is number_of_cells+1)",
    )
    lower_bound: list[float] | None = Field(
        default=None, description="Position of the node at the lower bound [m]"
    )
    upper_bound: list[float] | None = Field(
        default=None, description="Position of the node at the upper bound [m]"
    )
    lower_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    upper_boundary_conditions: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries, periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    # Per-axis scalar forms (resolved into the vector forms during validation)
    nx: int | None = Field(
        default=None, description="Number of cells along X (number of nodes=nx+1)"
    )
    ny: int | None = Field(
        default=None, description="Number of cells along Y (number of nodes=ny+1)"
    )
    nz: int | None = Field(
        default=None, description="Number of cells along Z (number of nodes=nz+1)"
    )
    xmin: float | None = Field(
        default=None, description="Position of first node along X [m]"
    )
    xmax: float | None = Field(
        default=None, description="Position of last node along X [m]"
    )
    ymin: float | None = Field(
        default=None, description="Position of first node along Y [m]"
    )
    ymax: float | None = Field(
        default=None, description="Position of last node along Y [m]"
    )
    zmin: float | None = Field(
        default=None, description="Position of first node along Z [m]"
    )
    zmax: float | None = Field(
        default=None, description="Position of last node along Z [m]"
    )
    bc_xmin: str | None = Field(
        default=None,
        description="Boundary condition at min X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_xmax: str | None = Field(
        default=None,
        description="Boundary condition at max X: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_ymin: str | None = Field(
        default=None,
        description="Boundary condition at min Y: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_ymax: str | None = Field(
        default=None,
        description="Boundary condition at max Y: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_zmin: str | None = Field(
        default=None,
        description="Boundary condition at min Z: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    bc_zmax: str | None = Field(
        default=None,
        description="Boundary condition at max Z: One of periodic, open, dirichlet, absorbing_silver_mueller, or neumann",
    )
    moving_window_velocity: list[float] | None = Field(
        default=None, description="Moving frame velocity [m/s]"
    )
    refined_regions: list = Field(
        default_factory=list,
        description="List of refined regions, each element being a list of the format [level, lo, hi, refinement_factor], with level being the refinement level, with 1 being the first level of refinement, 2 being the second etc, lo and hi being vectors of length 3 specifying the extent of the region, and refinement_factor defaulting to [2,2,2] (relative to next lower level)",
    )
    lower_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle lower bound [m]"
    )
    upper_bound_particles: list[float] | None = Field(
        default=None, description="Position of particle upper bound [m]"
    )
    xmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along X [m]"
    )
    xmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along X [m]"
    )
    ymin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along Y [m]"
    )
    ymax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along Y [m]"
    )
    zmin_particles: float | None = Field(
        default=None, description="Position of min particle boundary along Z [m]"
    )
    zmax_particles: float | None = Field(
        default=None, description="Position of max particle boundary along Z [m]"
    )
    lower_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at lower boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    upper_boundary_conditions_particles: list[str] | None = Field(
        default=None,
        description="Conditions at upper boundaries for particles, periodic, absorbing, reflect or thermal",
    )
    bc_xmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min X for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_xmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max X for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_ymin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min Y for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_ymax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max Y for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_zmin_particles: str | None = Field(
        default=None,
        description="Boundary condition at min Z for particles: One of periodic, absorbing, reflect, thermal",
    )
    bc_zmax_particles: str | None = Field(
        default=None,
        description="Boundary condition at max Z for particles: One of periodic, absorbing, reflect, thermal",
    )
    guard_cells: list[int] | None = Field(
        default=None, description="Number of guard cells used along each direction"
    )
    pml_cells: list[int] | None = Field(
        default=None,
        description="Number of Perfectly Matched Layer (PML) cells along each direction",
    )

    @model_validator(mode="after")
    @resolve_once
    def _resolve_grid(self) -> Self:
        # Sanity check and init of input arguments related to grid parameters
        assert (self.number_of_cells is not None) or (
            self.nx is not None and self.ny is not None and self.nz is not None
        ), "Either number_of_cells or nx, ny, and nz must be specified"
        assert (self.lower_bound is not None) or (
            self.xmin is not None
                and self.ymin is not None
                and self.zmin is not None
        ), "Either lower_bound or xmin, ymin, and zmin must be specified"
        assert (self.upper_bound is not None) or (
            self.xmax is not None
                and self.ymax is not None
                and self.zmax is not None
        ), "Either upper_bound or xmax, ymax, and zmax must be specified"
        # Note: like for the other grids, both forms may be given (the vector is used). This
        # validation re-runs on later assignments and when the grid is passed to another
        # PICMI object, at which point both forms are always set.
        assert (self.lower_boundary_conditions is not None) or (
            self.bc_xmin is not None
                and self.bc_ymin is not None
                and self.bc_zmin is not None
        ), "Either lower_boundary_conditions or bc_xmin, bc_ymin, and bc_zmin must be specified"
        assert (self.upper_boundary_conditions is not None) or (
            self.bc_xmax is not None
                and self.bc_ymax is not None
                and self.bc_zmax is not None
        ), "Either upper_boundary_conditions or bc_xmax, bc_ymax, and bc_zmax must be specified"

        if self.number_of_cells is None:
            self.number_of_cells = [self.nx, self.ny, self.nz]
        if self.lower_bound is None:
            self.lower_bound = [self.xmin, self.ymin, self.zmin]
        if self.upper_bound is None:
            self.upper_bound = [self.xmax, self.ymax, self.zmax]
        if self.lower_boundary_conditions is None:
            self.lower_boundary_conditions = [self.bc_xmin, self.bc_ymin, self.bc_zmin]
        if self.upper_boundary_conditions is None:
            self.upper_boundary_conditions = [self.bc_xmax, self.bc_ymax, self.bc_zmax]

        # Sanity check and init of input arguments related to particle boundary parameters
        # By default, if not specified, particle boundary values are the same as field boundary values
        # By default, if not specified, particle boundary conditions are the same as field boundary conditions
        if self.lower_bound_particles is None:
            self.lower_bound_particles = _fill_per_axis(
                [self.xmin_particles, self.ymin_particles, self.zmin_particles],
                self.lower_bound,
            )
        if self.upper_bound_particles is None:
            self.upper_bound_particles = _fill_per_axis(
                [self.xmax_particles, self.ymax_particles, self.zmax_particles],
                self.upper_bound,
            )

        if self.lower_boundary_conditions_particles is None:
            self.lower_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmin_particles, self.bc_ymin_particles, self.bc_zmin_particles],
                self.lower_boundary_conditions,
            )
        if self.upper_boundary_conditions_particles is None:
            self.upper_boundary_conditions_particles = _fill_per_axis(
                [self.bc_xmax_particles, self.bc_ymax_particles, self.bc_zmax_particles],
                self.upper_boundary_conditions,
            )

        # Sanity check on number of arguments of vector quantities
        assert len(self.number_of_cells) == 3, "Wrong number of cells specified"
        assert len(self.lower_bound) == 3, "Wrong number of lower bounds specified"
        assert len(self.upper_bound) == 3, "Wrong number of upper bounds specified"
        assert len(self.lower_boundary_conditions) == 3, (
            "Wrong number of lower boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions) == 3, (
            "Wrong number of upper boundary conditions specified"
        )
        assert len(self.lower_bound_particles) == 3, (
            "Wrong number of particle lower bounds specified"
        )
        assert len(self.upper_bound_particles) == 3, (
            "Wrong number of particle upper bounds specified"
        )
        assert len(self.lower_boundary_conditions_particles) == 3, (
            "Wrong number of particle lower boundary conditions specified"
        )
        assert len(self.upper_boundary_conditions_particles) == 3, (
            "Wrong number of particle upper boundary conditions specified"
        )

        for region in self.refined_regions:
            if len(region) == 3:
                region.append([2, 2, 2])
            assert len(region[1]) == 3, (
                "The lo extent of the refined region must be a vector of length 3"
            )
            assert len(region[2]) == 3, (
                "The hi extent of the refined region must be a vector of length 3"
            )
            assert len(region[3]) == 3, (
                "The refinement factor of the refined region must be a vector of length 3"
            )

        return self

    def add_refined_region(self, level, lo, hi, refinement_factor=[2, 2, 2]):
        """Add a refined region.

        Parameters
        ----------
        level: integer
            The refinement level, with 1 being the first level of refinement, 2 being the second etc.

        lo, hi: vectors of floats
            Each is a vector of length 3 specifying the extent of the region

        refinement_factor: vector of integers, optional
            Defaulting to [2,2,2] (relative to next lower level)
        """
        self.refined_regions.append([level, lo, hi, refinement_factor])


PICMI_AnyGrid = (
    PICMI_CylindricalGrid
    | PICMI_Cartesian1DGrid
    | PICMI_Cartesian2DGrid
    | PICMI_Cartesian3DGrid
)

_ElectromagneticSolverMethod = Literal[
    "Yee", "CKC", "Lehe", "PSTD", "PSATD", "GPSTD", "DS", "ECT"
]


class PICMI_ElectromagneticSolver(_PICMIModel):
    """
    Electromagnetic field solver.

    The advance method used to solve Maxwell's equations. The default method is code dependent.

    Method options:

    - 'Yee': standard solver using the staggered Yee grid (https://doi.org/10.1109/TAP.1966.1138693)
    - 'CKC': solver with the extended Cole-Karkkainen-Cowan stencil with better dispersion properties (https://doi.org/10.1103/PhysRevSTAB.16.041303)
    - 'Lehe': CKC-style solver with modified dispersion (https://doi.org/10.1103/PhysRevSTAB.16.021301)
    - 'PSTD': Spectral solver with finite difference in time domain, e.g., Q. H. Liu, Letters 15 (3) (1997) 158–165
    - 'PSATD': Spectral solver with analytic in time domain (https://doi.org/10.1016/j.jcp.2013.03.010)
    - 'DS': Directional Splitting after Yasuhiko Sentoku (https://doi.org/10.1140/epjd/e2014-50162-y)
    - 'ECT': Enlarged Cell Technique solver, allowing internal conductors (https://doi.org/10.1109/APS.2005.1551259)
    """

    # Retained for backwards compatibility reasons.
    # The type annotation of `method` is the ground-truth.
    methods_list: ClassVar[list[str]] = list(get_args(_ElectromagneticSolverMethod))

    grid: PICMI_AnyGrid = Field(description="Grid object for the diagnostic")
    method: _ElectromagneticSolverMethod | None = Field(
        default=None,
        description="The advance method use to solve Maxwell's equations. The default method is code dependent.",
    )
    stencil_order: Sequence[int] | None = Field(
        default=None, description="Order of stencil for each axis (-1=infinite)"
    )
    cfl: float | None = Field(
        default=None, description="Fraction of the Courant-Friedrich-Lewy criteria [1]"
    )
    source_smoother: PICMI_BinomialSmoother | None = Field(
        default=None, description="Smoother object to apply to the sources"
    )
    field_smoother: PICMI_BinomialSmoother | None = Field(
        default=None, description="Smoother object to apply to the fields"
    )
    subcycling: int | None = Field(
        default=None, description="Level of subcycling for the GPSTD solver"
    )
    galilean_velocity: Sequence[float] | None = Field(
        default=None, description="Velocity of Galilean reference frame [m/s]"
    )
    divE_cleaning: bool | None = Field(
        default=None, description="Solver uses div(E) cleaning if True"
    )
    divB_cleaning: bool | None = Field(
        default=None, description="Solver uses div(B) cleaning if True"
    )
    pml_divE_cleaning: bool | None = Field(
        default=None, description="Solver uses div(E) cleaning in the PML if True"
    )
    pml_divB_cleaning: bool | None = Field(
        default=None, description="Solver uses div(B) cleaning in the PML if True"
    )


class PICMI_ElectrostaticSolver(_ClassWithInit):
    """
    Electrostatic field solver

    Parameters
    ----------
    grid: grid instance
        Grid object for the diagnostic

    method: string
        One of 'FFT', or 'Multigrid'

    required_precision: float, optional
        Level of precision required for iterative solvers

    maximum_iterations: integer, optional
        Maximum number of iterations for iterative solvers
    """

    methods_list = ["FFT", "Multigrid"]

    def __init__(
        self, grid, method=None, required_precision=None, maximum_iterations=None, **kw
    ):
        assert method is None or method in PICMI_ElectrostaticSolver.methods_list, (
            Exception(
                "method must be one of "
                + ", ".join(PICMI_ElectrostaticSolver.methods_list)
            )
        )

        self.grid = grid
        self.method = method
        self.required_precision = required_precision
        self.maximum_iterations = maximum_iterations

        self.handle_init(kw)


class PICMI_MagnetostaticSolver(_ClassWithInit):
    """
    Magnetostatic field solver

    Parameters
    ----------
    grid: grid instance
        Grid object for the diagnostic

    method: string
        One of 'FFT', or 'Multigrid'
    """

    methods_list = ["FFT", "Multigrid"]

    def __init__(self, grid, method=None, **kw):
        assert method is None or method in PICMI_MagnetostaticSolver.methods_list, (
            Exception(
                "method must be one of "
                + ", ".join(PICMI_MagnetostaticSolver.methods_list)
            )
        )

        self.grid = grid
        self.method = method

        self.handle_init(kw)
