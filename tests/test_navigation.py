import numpy as np

from morpheus_policy.navigation import NavigationConfig, RouteController


def test_route_controller_settles_then_emits_bounded_body_command_and_stops_at_goal():
    config = NavigationConfig(speed_m_s=0.3, max_yaw_rate=0.6, acceleration_m_s2=0.25, settle_s=0.5)
    controller = RouteController([[0.0, 0.0], [1.0, 0.0]], config)

    assert np.array_equal(controller.command([0.0, 0.0], 0.0, 0.49, 0.1), np.zeros(3))
    command = controller.command([0.0, 0.0], 0.2, 0.6, 0.1)

    assert 0.0 <= command[0] <= config.speed_m_s
    assert command[0] <= config.acceleration_m_s2 * 0.1
    assert command[1] == 0.0
    assert abs(command[2]) <= config.max_yaw_rate
    assert np.array_equal(controller.command([0.9, 0.0], 0.0, 0.7, 0.1), np.zeros(3))
