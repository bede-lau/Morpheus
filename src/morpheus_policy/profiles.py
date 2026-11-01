"""Navigation envelopes for viewer profiles; these are not robot dynamics models."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NavigationProfile:
    id: str
    radius_m: float
    body_height_m: float
    max_speed_m_s: float = 0.30
    max_yaw_rate: float = 0.60
    max_lateral_speed_m_s: float = 1e-6
    mobile: bool = True
    physics_robot_id: str | None = None

    def contract(self):
        return {**asdict(self), "geometry_status": "assumed_conservative_footprint",
                "training_model": "reduced_order_static_footprint"}


# A registry makes additions explicit without selecting behavior by robot name
# in the learner, planner, or evaluator. These sizes are demo assumptions.
NAVIGATION_PROFILES = {
    "g1": NavigationProfile("g1", .35, 1.40, physics_robot_id="unitree_g1_12dof"),
    "stretch": NavigationProfile("stretch", .40, 1.50),
    "go2": NavigationProfile("go2", .40, .75),
    "soarm": NavigationProfile("soarm", .25, .65, mobile=False),
}


def navigation_profile(profile_id):
    try:
        return NAVIGATION_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown navigation profile: {profile_id}") from error
