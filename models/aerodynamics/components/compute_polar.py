"""Computation of CL and CD for whole aircraft."""
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
from ..constants import PolarType, SERVICE_POLAR


@RegisterSubmodel(SERVICE_POLAR, "fastoad.submodel.aerodynamics.polar.legacy")
class ComputePolar(om.ExplicitComponent):
    """Computation of CL and CD for whole aircraft."""

    def initialize(self):
        self.options.declare("polar_type", default=PolarType.HIGH_SPEED, types=PolarType)

    def setup(self):
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:k", val=np.nan)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:offset", val=np.nan)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k", val=np.nan)
        self.add_input("tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset", val=np.nan)

        if self.options["polar_type"] != PolarType.HIGH_SPEED:
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CL", shape_by_conn=True, val=np.nan
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CD0", shape_by_conn=True, val=np.nan
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:CD:trim", shape_by_conn=True, val=np.nan
            )
            self.add_input(
                "data:aerodynamics:aircraft:low_speed:induced_drag_coefficient", val=np.nan
            )

            if self.options["polar_type"] == PolarType.TAKEOFF:
                self.add_input("data:aerodynamics:high_lift_devices:takeoff:CL", val=np.nan)
                self.add_input("data:aerodynamics:high_lift_devices:takeoff:CD", val=np.nan)
                self.add_input("data:aerodynamics:blown_wing_aero:CL", val=np.nan)
                self.add_input("data:aerodynamics:blown_wing_aero:CD", val=np.nan)
                self.add_input("data:aerodynamics:blown_wing_aero:bool_version", val=1)
                self.add_output(
                    "data:aerodynamics:blown_wing_aero:CL_V2",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )  # check
                self.add_output(
                    "data:aerodynamics:blown_wing_aero:CD_V2",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )  # check
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CL",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:takeoff:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )

            elif self.options["polar_type"] == PolarType.LANDING:
                self.add_input("data:aerodynamics:high_lift_devices:landing:CL", val=np.nan)
                self.add_input("data:aerodynamics:high_lift_devices:landing:CD", val=np.nan)
                self.add_output(
                    "data:aerodynamics:landing:CL",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CL",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
                self.add_output(
                    "data:aerodynamics:aircraft:landing:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
            elif self.options["polar_type"] == PolarType.LOW_SPEED:
                self.add_output(
                    "data:aerodynamics:aircraft:low_speed:CD",
                    copy_shape="data:aerodynamics:aircraft:low_speed:CL",
                )
            else:
                raise AttributeError(f'Unknown polar type: {self.options["polar_type"]}')

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            self.add_input("data:aerodynamics:aircraft:cruise:CL", shape_by_conn=True, val=np.nan)
            self.add_input("data:aerodynamics:aircraft:cruise:CD0", shape_by_conn=True, val=np.nan)
            self.add_input(
                "data:aerodynamics:aircraft:cruise:CD:trim", shape_by_conn=True, val=np.nan
            )
            self.add_input(
                "data:aerodynamics:aircraft:cruise:CD:compressibility",
                shape_by_conn=True,
                val=np.nan,
            )
            self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", val=np.nan)

            self.add_output(
                "data:aerodynamics:aircraft:cruise:CD",
                copy_shape="data:aerodynamics:aircraft:cruise:CL",
            )
            self.add_output("data:aerodynamics:aircraft:cruise:L_D_max")
            self.add_output("data:aerodynamics:aircraft:cruise:optimal_CL")
            self.add_output("data:aerodynamics:aircraft:cruise:optimal_CD")

        else:
            raise AttributeError(f'Unknown polar type: {self.options["polar_type"]}')

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        k_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:k"]
        offset_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:offset"]
        k_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k"]
        offset_winglet_cd = inputs["tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset"]

        
        if self.options["polar_type"] != PolarType.HIGH_SPEED:
            cl = inputs["data:aerodynamics:aircraft:low_speed:CL"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            cd_trim = inputs["data:aerodynamics:aircraft:low_speed:CD:trim"]
            cd_c = 0.0
            coef_k = inputs["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
            if self.options["polar_type"] == PolarType.TAKEOFF:
                delta_cl_hl = inputs["data:aerodynamics:high_lift_devices:takeoff:CL"]
                delta_cd_hl = inputs["data:aerodynamics:high_lift_devices:takeoff:CD"]
                bool_version = inputs["data:aerodynamics:blown_wing_aero:bool_version"]
                if bool_version > 0 : # Version 1 : Theoric model
                    delta_cl_bw = inputs["data:aerodynamics:blown_wing_aero:CL"]
                    delta_cd_bw = inputs["data:aerodynamics:blown_wing_aero:CD"]
            elif self.options["polar_type"] == PolarType.LANDING:
                delta_cl_hl = inputs["data:aerodynamics:high_lift_devices:landing:CL"]
                delta_cd_hl = inputs["data:aerodynamics:high_lift_devices:landing:CD"]
                delta_cl_bw = 0.0
                delta_cd_bw = 0.0
            elif self.options["polar_type"] == PolarType.LOW_SPEED:
                delta_cl_hl = 0.0
                delta_cd_hl = 0.0
                delta_cl_bw = 0.0
                delta_cd_bw = 0.0
            else:
                raise AttributeError(f'Unknown polar type: {self.options["polar_type"]}')

        elif self.options["polar_type"] == PolarType.HIGH_SPEED:
            cl = inputs["data:aerodynamics:aircraft:cruise:CL"]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            cd_trim = inputs["data:aerodynamics:aircraft:cruise:CD:trim"]
            cd_c = inputs["data:aerodynamics:aircraft:cruise:CD:compressibility"]
            coef_k = inputs["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
            delta_cl_hl = 0.0
            delta_cd_hl = 0.0
            delta_cl_bw = 0.0
            delta_cd_bw = 0.0
        else:
            raise AttributeError(f'Unknown polar type: {self.options["polar_type"]}')

        cl = cl + delta_cl_hl
        cd = (
            cd0 + cd_c + cd_trim + coef_k * cl ** 2 * k_winglet_cd + offset_winglet_cd + delta_cd_hl
        ) * k_cd + offset_cd

        if self.options["polar_type"] == PolarType.TAKEOFF:
            if bool_version > 0 :
                cl = cl + delta_cl_bw
                cd = cd + delta_cd_bw * k_cd
            else:
                delta_cl_bw, delta_cd_bw = blown_wing_aero_v2(cl, cd)
                cl = cl + delta_cl_bw
                cd = cd + delta_cd_bw * k_cd

        if self.options["polar_type"] == PolarType.LOW_SPEED:
            outputs["data:aerodynamics:aircraft:low_speed:CD"] = cd
        elif self.options["polar_type"] == PolarType.TAKEOFF:
            outputs["data:aerodynamics:aircraft:takeoff:CL"] = cl
            outputs["data:aerodynamics:aircraft:takeoff:CD"] = cd

            # Check
            if bool_version > 0:
                outputs["data:aerodynamics:blown_wing_aero:CL_V2"] = cl
                outputs["data:aerodynamics:blown_wing_aero:CD_V2"] = cd
            else:
                outputs["data:aerodynamics:blown_wing_aero:CL_V2"] = delta_cl_bw
                outputs["data:aerodynamics:blown_wing_aero:CD_V2"] = delta_cd_bw
        elif self.options["polar_type"] == PolarType.LANDING:
            outputs["data:aerodynamics:aircraft:landing:CL"] = cl
            outputs["data:aerodynamics:aircraft:landing:CD"] = cd
        else:
            outputs["data:aerodynamics:aircraft:cruise:CD"] = cd

            Cl_opt, Cd_opt = get_optimum_ClCd(np.array([cd, cl]))[0:2]
            outputs["data:aerodynamics:aircraft:cruise:L_D_max"] = Cl_opt / Cd_opt
            outputs["data:aerodynamics:aircraft:cruise:optimal_CL"] = Cl_opt
            outputs["data:aerodynamics:aircraft:cruise:optimal_CD"] = Cd_opt


def get_optimum_ClCd(ClCd):
    lift_drag_ratio = ClCd[1, :] / ClCd[0, :]
    optimum_index = np.argmax(lift_drag_ratio)

    optimum_Cz = ClCd[1][optimum_index]
    optimum_Cd = ClCd[0][optimum_index]

    return optimum_Cz, optimum_Cd

def blown_wing_aero_v2(cl,cd):
    """ Compute the increment in lift and drag
    VERSION 2 Blown Wing Aero : Model based on data"""
    delta_cl = cl * 0.04
    delta_cd = - cd * 0.01
    return delta_cl, delta_cd
