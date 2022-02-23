"""
Computation distributed propulsion
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

import openmdao.api as om
import numpy as np

from fastoad.module_management.constants import ModelDomain
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.model_base import Atmosphere

from .exceptions import FastDistrPropulsionError


@RegisterOpenMDAOSystem(
    "MYfastoad.propulsion.prop_distr", domain=ModelDomain.PROPULSION,
)
class ComputePropDistrib(om.Group):
    """
    Computes areas of vertical and horizontal tail.

    - Horizontal tail area is computed so it can balance pitching moment of
      aircraft at rotation speed.
    - Vertical tail area is computed so aircraft can have the CNbeta in cruise
      conditions
    """

    def setup(self):
        self.add_subsystem("prop_distr", PROP_DISTRIB(), promotes=["*"])

class PROP_DISTRIB(om.ExplicitComponent):

    def setup(self):
        self.add_input("tuning:propeller:beta_pro", val=np.nan)
        self.add_input("data:propulsion:MTO_thrust", val=np.nan, units="N")
        self.add_input("tuning:propeller:count", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:geometry:propulsion:propeller:diameter", units="m")
        self.add_output("data:propulsion:propeller:thrust_prop", units="N")
        self.add_output("data:aerodynamics:propulsion:propeller:speed_ejected", units="m/s")
        self.add_output("data:aerodynamics:propulsion:propeller:speed_disk", units="m/s")
        self.add_output("data:propulsion:propeller:power_tot", units="W")

    def setup_partials(self):
            self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        beta_pro = inputs["tuning:propeller:beta_pro"]
        MTO = inputs["data:propulsion:MTO_thrust"]
        n_prop = inputs["tuning:propeller:count"]
        wing_span = inputs["data:geometry:wing:span"]

        # Sizing at take-off
        takeoff_mach = 0.3
        alt = 0
        C_t = (4.27e-02 + 1.44e-01 * beta_pro)  # Thrust coef with T=C_T.rho.n^2.D^4 - 0.8 for de-rating of APC catalog
        C_p = -1.48e-03 + 9.72e-02 * beta_pro  # Power coef with P=C_p.rho.n^3.D^5
        # Propeller selection with take-off scenario
        atm = Atmosphere(alt)
        speed_0 = takeoff_mach * atm.speed_of_sound
        rho = atm.density
        T_prop = MTO/n_prop
        ND_max = 0.7 * atm.speed_of_sound
        D_pro = (T_prop / (C_t * rho * (ND_max) ** 2.0)) ** 0.5  # [m] Propeller diameter
        n_pro_to = ND_max / D_pro  # [Hz] Propeller speed
        # Omega_pro_to = n_pro_to * 2 * np.pi  # [rad/s] Propeller speed
        P_pro_to = (C_p * rho * n_pro_to ** 3.0 * D_pro ** 5.0)  # [W] Power per propeller
        Power_tot = P_pro_to * n_prop
        Area_disk = np.pi * (D_pro/2)**2
        Speed_ejec = np.sqrt(2 * T_prop /(rho*Area_disk) + speed_0**2)
        Speed_disk = 0.5*(Speed_ejec + speed_0)

        # Error if span propellers bigger than wing_span
        span_pro = D_pro * n_prop/2 * 1.1
        if span_pro > (wing_span/2):
                raise FastDistrPropulsionError(
                    "Error: there is no enough space for propellers on the wing."
                )

        outputs["data:geometry:propulsion:propeller:diameter"] = D_pro
        outputs["data:propulsion:propeller:thrust_prop"] = T_prop
        outputs["data:propulsion:propeller:power_tot"] = Power_tot
        outputs["data:aerodynamics:propulsion:propeller:speed_ejected"] = Speed_ejec
        outputs["data:aerodynamics:propulsion:propeller:speed_disk"] = Speed_disk