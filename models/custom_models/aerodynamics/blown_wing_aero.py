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
from fastoad.module_management.constants import ModelDomain
from fastoad.model_base import Atmosphere
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem, RegisterSubmodel
# from ..constants import SERVICE_BLOWN_WING_AERO


@RegisterOpenMDAOSystem("fastoad.custom_models.aerodynamics.blown_wing_aero.legacy", domain=ModelDomain.AERODYNAMICS)
# HAVE TO BE A SUBMODEL
class ComputeDeltaBlownWing(om.ExplicitComponent):
    """
    Provides lift and drag increments due to blown wing effect
    """

    def initialize(self):
        self.options.declare("landing_flag", default=False, types=bool)

    def setup(self):
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_input("data:aerodynamics:aircraft:takeoff:mach", val=np.nan)
        # self.add_input("data:aerodynamics:propulsion:propeller:speed_ejected", val=np.nan, units="m/s")
        # self.add_input("data:propulsion:propeller:thrust_prop", val=np.nan, units="N")
        # self.add_input("data:geometry:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_output("data:aerodynamics:blown_wing_aero:CL")
        self.add_output("data:aerodynamics:blown_wing_aero:CD")

        # Check variables :
        self.add_output("data:aerodynamics:blown_wing_aero:delta_mass_flow")
        self.add_output("data:aerodynamics:blown_wing_aero:air_speed_s")
        self.add_output("data:aerodynamics:blown_wing_aero:delta_lift_s")
        self.add_output("data:aerodynamics:blown_wing_aero:delta_drag_s")
        self.add_output("data:aerodynamics:blown_wing_aero:wet_area_wing_s")
        self.add_output("data:aerodynamics:blown_wing_aero:air_force_s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        wing_area = inputs["data:geometry:wing:area"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        span = inputs["data:geometry:wing:span"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        mach = inputs["data:aerodynamics:aircraft:takeoff:mach"]
        # speed_ejected = inputs["data:aerodynamics:propulsion:propeller:speed_ejected"]
        # thrust_prop = inputs["data:propulsion:propeller:thrust_prop"]
        # prop_diameter = inputs["data:geometry:propulsion:propeller:diameter"]

        speed_ejected = 1000.
        thrust_prop = 82000.
        prop_diameter = 2.

        k_factor = 0.000000004 * thrust_prop * n_engines # Randomly choosen for instance
        angle_streamtubes = np.pi / 12. # Randomly choosen for instance

        # Compute true airspeed and air density
        alt = 0.
        atm = Atmosphere(alt)
        speed_0 = mach * atm.speed_of_sound
        rho = atm.density

        # Compute mass flow (only the added contribution of the distributed propulsion on wings) providing by all engines
        delta_mass_flow = n_engines * rho * (speed_ejected - speed_0) * prop_diameter ** 2 / 4.

        # Compute the air speed in streamtubes on wing
        air_speed_s = (1. - k_factor) * speed_ejected

        # Compute the contribution in lift (streamtube and added contribution only)
        delta_lift_s = delta_mass_flow * (air_speed_s * np.cos(angle_streamtubes) - speed_0)

        # Compute the contribution in drag (streamtube and added contribution only)
        delta_drag_s = delta_mass_flow * air_speed_s * np.sin(angle_streamtubes)

        # Compute wetted area (streamtube and added contribution only)
        wet_area_wing = 2 * (wing_area - width_max * l2_wing)
        wet_area_wing_s = (wet_area_wing * prop_diameter / (2 * span) * n_engines)

        # Compute air force in the propellers streamtube
        air_force_s = rho * air_speed_s ** 2 * wet_area_wing_s

        outputs["data:aerodynamics:blown_wing_aero:CL"] = delta_lift_s / air_force_s
        outputs["data:aerodynamics:blown_wing_aero:CD"] = delta_drag_s / air_force_s

        # Check
        outputs["data:aerodynamics:blown_wing_aero:delta_mass_flow"] = delta_mass_flow
        outputs["data:aerodynamics:blown_wing_aero:air_speed_s"] = air_speed_s
        outputs["data:aerodynamics:blown_wing_aero:delta_lift_s"] = delta_lift_s
        outputs["data:aerodynamics:blown_wing_aero:delta_drag_s"] = delta_drag_s
        outputs["data:aerodynamics:blown_wing_aero:wet_area_wing_s"] = wet_area_wing_s
        outputs["data:aerodynamics:blown_wing_aero:air_force_s"] = air_force_s