"""
Estimation of vertical tail area
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
from fastoad.model_base import Atmosphere


class ComputeVTArea(om.ExplicitComponent):
    """
    Computes area of vertical tail plane

    Area is computed to fulfill lateral stability requirement (with the most aft CG)
    as stated in :cite:raymer:1992.
    """

    def setup(self):
        self.add_input("data:TLAR:cruise_mach", val=np.nan)
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cruise:CnBeta", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:cruise:CL_alpha", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("tuning:propeller:count", val=np.nan)
        self.add_input("data:propulsion:MTO_thrust", val=np.nan, units="N")
        self.add_input("data:geometry:propulsion:propeller:y1", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:wetted_area", units="m**2", ref=100.0)
        self.add_output("data:geometry:vertical_tail:area", units="m**2", ref=50.0)
        self.add_output("data:aerodynamics:vertical_tail:cruise:CnBeta")
        self.add_output("data:aerodynamics:vertical_tail:cruise:CnBeta_mot")
        self.add_output("data:aerodynamics:vertical_tail:cruise:CnBeta_goal")

    def setup_partials(self):
        self.declare_partials("data:geometry:vertical_tail:wetted_area", "*", method="fd")
        self.declare_partials("data:geometry:vertical_tail:area", "*", method="fd")
        self.declare_partials("data:aerodynamics:vertical_tail:cruise:CnBeta", "*", method="fd")
        self.declare_partials("data:aerodynamics:vertical_tail:cruise:CnBeta_mot", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # pylint: disable=too-many-locals  # needed for clarity

        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_mac_position = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        cn_beta_fuselage = inputs["data:aerodynamics:fuselage:cruise:CnBeta"]
        cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]
        cruise_mach = inputs["data:TLAR:cruise_mach"]
        # This one is the distance between the 25% MAC points
        wing_htp_distance = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        n_engines = inputs["tuning:propeller:count"]
        thrust_sl = inputs["data:propulsion:MTO_thrust"]
        y1 = inputs["data:geometry:propulsion:propeller:y1"]

        # Matches suggested goal by Raymer, Fig 16.20
        cn_beta_goal = 0.0569 - 0.01694 * cruise_mach + 0.15904 * cruise_mach ** 2

        atm = Atmosphere(altitude)
        speed = cruise_mach * atm.speed_of_sound
        rho = atm.density
        # dynamic pressure
        q = 0.5 * rho * speed**2 * wing_area * l0_wing
        # Cn_beta produced by distributed propulsion
        # Initialize the Cn_beta_prop to enter in the loop
        cn_beta_prop = -0.001
        # Initialize the coefficient theta that establish the thrust distribution
        theta = 1
        thrust_cr = 0.65 * thrust_sl
        thrust_cr_i = thrust_cr / n_engines

        while cn_beta_prop < (0.2 * cn_beta_goal) and theta < 2:
            # Define a factor k to ensure that thrust on the internal propeller is less than MTO_thrust
            # considering that the thrust in cruise is equal to 0.65 of MTO_thrust
            k_tmax = 1.93 / (theta * n_engines / 4 - 1)
            # Define the variable gamma referred to the position of the last propeller
            gamma = n_engines * y1 / 2
            cn_beta_prop = (k_tmax * thrust_cr_i * n_engines * (theta * gamma * (gamma + 1) / 8 -
                                                                (gamma + 1)*(2*gamma + 1) / 12)) / q
            theta = theta + 0.01

        required_cnbeta_vtp = cn_beta_goal - cn_beta_fuselage - cn_beta_prop
        # required_cnbeta_vtp = cn_beta_goal - cn_beta_fuselage
        distance_to_cg = wing_htp_distance + 0.25 * l0_wing - cg_mac_position * l0_wing
        vt_area = required_cnbeta_vtp / (distance_to_cg / wing_area / span * cl_alpha_vt)
        wet_vt_area = 2.1 * vt_area

        outputs["data:geometry:vertical_tail:wetted_area"] = wet_vt_area
        outputs["data:geometry:vertical_tail:area"] = vt_area
        outputs["data:aerodynamics:vertical_tail:cruise:CnBeta"] = required_cnbeta_vtp
        outputs["data:aerodynamics:vertical_tail:cruise:CnBeta_mot"] = cn_beta_prop
        outputs["data:aerodynamics:vertical_tail:cruise:CnBeta_goal"] = cn_beta_goal
