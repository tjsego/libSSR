from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator

from mkstd.standards import (
    Hdf5Standard,
    JsonStandard,
    XmlStandard,
    YamlStandard,
)
from mkstd.types.array import get_array_type

# Stage 1: define a data model, then make standards from it.


class EFECTReport(BaseModel):
    """Data standard for EFECT reports."""

    efect_level: int = Field(default=None, ge=0)
    """EFECT level"""
    efect_version: int = Field(default=None, ge=0)
    """EFECT version"""

    variable_names: list[str]
    """Variable names"""
    simulation_times: get_array_type(
        dtype=np.float64, dimensions=1, strict_dtype=True
    )
    """Simulation times"""
    sample_size: int = Field(default=None, ge=1)
    """Sample size"""

    ecf_evals: get_array_type(
        dtype=np.float64, dimensions=4, strict_dtype=True
    )
    """ECF evaluations

    - dim 0 is simulation time
    - dim 1 is variable name
    - dim 2 is evaluation index
    - dim 3 is real/imaginary component
    """

    ecf_tval: get_array_type(dtype=np.float64, dimensions=2, strict_dtype=True)
    """ECF evaluation final value

    - dim 0 is simulation time
    - dim 1 is variable name
    """
    ecf_nval: int = Field(None, ge=1)
    """Number of evaluations per combination of simulation time and variable"""

    error_metric_mean: float
    """Error metric mean, from test for reproducibility"""
    error_metric_stdev: float = Field(ge=0)
    """Error metric mean, from test for reproducibility"""

    sig_figs: int = Field(ge=1)
    """Significant figures of sample data"""

    @model_validator(mode="after")
    def ensure_array_dimensions(self: EFECTReport) -> EFECTReport:
        """Ensure that dependencies of array dimensions are satisfied."""
        # Keys are used in the expectation message.
        attr_measures = {
            "the number of ": lambda x: len(x),
            "": lambda x: x,
        }
        # These expectations have three elements:
        # 1: array name and dimension
        # 2: how to measure the expected value (a key of `attr_measures`)
        # 3: the object that provides the expected value
        expectations = [
            (
                ("ecf_evals", 0),
                "the number of ",
                "simulation_times",
            ),
            (
                ("ecf_evals", 1),
                "the number of ",
                "variable_names",
            ),
            (
                ("ecf_evals", 2),
                "",
                "ecf_nval",
            ),
            (
                ("ecf_evals", 3),
                "",
                2,
            ),
            (
                ("ecf_tval", 0),
                "the number of ",
                "simulation_times",
            ),
            (
                ("ecf_tval", 1),
                "the number of ",
                "variable_names",
            ),
        ]
        for (attr1, dim), attr0_kind, attr0 in expectations:
            test_value = getattr(self, attr1).shape[dim]
            if isinstance(attr0, str):
                expected_value = attr_measures[attr0_kind](
                    getattr(self, attr0)
                )
                current_expected_value = f" (currently `{expected_value}`)"
            else:
                expected_value = attr0
                current_expected_value = ""
            if test_value != expected_value:
                raise ValueError(
                    f"Dimension `{dim}` of `{attr1}` (currently "
                    f"`{test_value}`) must equal {attr0_kind}`{attr0}`"
                    f"{current_expected_value}."
                )
        return self


hdf5_standard = Hdf5Standard(model=EFECTReport)
hdf5_standard.save_schema("standards/schema_hdf5.json")

json_standard = JsonStandard(model=EFECTReport)
json_standard.save_schema("standards/schema.json")

xml_standard = XmlStandard(model=EFECTReport)
xml_standard.save_schema("standards/schema.xsd")

yaml_standard = YamlStandard(model=EFECTReport)
yaml_standard.save_schema("standards/schema.yaml")

# # Stage 2: use standards to validate/import/export data.
# # Here, data are validated via `mkstd`. Third-party validators
# # can also be used.
# data = EFECTReport.parse_obj(
#     {
#         "efect_level": 1,
#         "efect_version": 3,
#         "variable_names": ["v1", "v2", "v3"],
#         "simulation_times": np.linspace(0, 10, 4),
#         "sample_size": 100000,
#         "ecf_evals": np.zeros((4, 3, 5, 2)),
#         "ecf_tval": np.ones((4, 3)),
#         "ecf_nval": 5,
#         "error_metric_mean": 0.5,
#         "error_metric_stdev": 0.2,
#         "sig_figs": 5,
#     }
# )
# 
# xml_standard.save_data(data=data, filename="data/data.xml")
# data_xml = xml_standard.load_data(filename="data/data.xml")
# 
# json_standard.save_data(data=data, filename="data/data.json")
# data_json = json_standard.load_data(filename="data/data.json")
# 
# yaml_standard.save_data(data=data, filename="data/data.yaml")
# data_yaml = yaml_standard.load_data(filename="data/data.yaml")
# 
# hdf5_standard.save_data(data=data, filename="data/data.hdf5")
# data_hdf5 = hdf5_standard.load_data(filename="data/data.hdf5")
