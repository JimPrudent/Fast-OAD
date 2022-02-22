"""
Computation of lift and drag increments due to blown wing service
"""
#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2021 ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel
from ..constants import SERVICE_BLOWN_WING_AERO


@RegisterSubmodel(SERVICE_BLOWN_WING_AERO, "fastoad.submodel.aerodynamics.blown_wing_aero.legacy")
class ComputeDeltaBlownWing(om.ExplicitComponent):
    """
    Provides lift and drag increments due to blown wing effect
    """

    def initialize(self):
        self.options.declare("landing_flag", default=False, types=bool)

    def setup(self):
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_output("data:aerodynamics:blown_wing_aero:CL")
        self.add_output("data:aerodynamics:blown_wing_aero:CD")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        span_ratio = inputs["data:geometry:flap:span_ratio"]
        outputs["data:aerodynamics:blown_wing_aero:CL"] = 2. # Just a test, it's not true
        outputs["data:aerodynamics:blown_wing_aero:CD"] = 2. # Just a test, it's not true