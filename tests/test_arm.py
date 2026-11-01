import mujoco

from morpheus_policy.arm import solve_reach


_TWO_LINK_MJCF = """
<mujoco model="two_link_fixture">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link_1">
      <joint name="joint_1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02"/>
      <body name="link_2" pos="0.4 0 0">
        <joint name="joint_2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02"/>
        <site name="tool" pos="0.4 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _fixture():
    model = mujoco.MjModel.from_xml_string(_TWO_LINK_MJCF)
    return model, mujoco.MjData(model)


def test_generic_arm_fixture_reports_reachable_kinematic_target_without_robot_claim():
    model, data = _fixture()
    result = solve_reach(model, data, ["joint_1", "joint_2"], "tool", [0.4, 0.4, 0.0])

    assert result["reached"] is True
    assert result["target_error_m"] <= 0.01
    assert result["collision_verified"] is False
    assert result["hardware_readiness"] == "UNVERIFIED"


def test_generic_arm_fixture_reports_unreachable_target():
    model, data = _fixture()
    result = solve_reach(model, data, ["joint_1", "joint_2"], "tool", [4.0, 4.0, 0.0], max_iterations=80)

    assert result["reached"] is False
    assert result["target_error_m"] > 0.01
