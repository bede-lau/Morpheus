"""Physical manipulation extensions for Morpheus's MuJoCo kitchen fixture."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

GO2_MJCF_SHA256 = "50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a"
G1_HANDS_MJCF_SHA256 = "a6d88c9efde50e9715c71e590ce1a8d3515a0099df7f36e62add8204dae758ad"
GO2_ARM_JOINTS = ("kit_lift", "kit_extension", "kit_left_jaw", "kit_right_jaw")


def box(body, name, size, position, mass, color):
    """Separate contact geometry from the meshes used in the visual GLB."""
    attrs = dict(type="box",size=" ".join(map(str,size)),pos=" ".join(map(str,position)),rgba=color)
    ET.SubElement(body,"geom",name=name,group="3",mass=str(mass),friction=".9 .015 .002",condim="4",**attrs)
    ET.SubElement(body,"geom",name=name+"_visual",group="2",mass="0",contype="0",conaffinity="0",**attrs)


def go2_arm(xml):
    """Attach a simulated prismatic mast, forward slide, and two powered jaws.

    All masses and collision dimensions participate in the same Go2 forward
    dynamics as the legs. The attachment is a prototype, with no hardware claim.
    """
    if hashlib.sha256(Path(xml).read_bytes()).hexdigest()!=GO2_MJCF_SHA256:
        raise ValueError("Expected the pinned Go2 Menagerie model")
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir",str(Path(xml).parent/"assets"))
    base = root.find("./worldbody/body")
    mast = ET.SubElement(base,"body",name="kit_mast",pos=".055 0 .095")
    box(mast,"kit_column",(.031,.039,.23),(0,0,.23),.31,".76 .54 .15 1")
    lift = ET.SubElement(mast,"body",name="kit_carriage",pos="0 0 .12")
    ET.SubElement(lift,"joint",name="kit_lift",type="slide",axis="0 0 1",range="0 .49",damping="5")
    box(lift,"kit_carriage_box",(.044,.055,.036),(0,0,0),.10,".63 .67 .70 1")
    arm = ET.SubElement(lift,"body",name="kit_slide",pos=".126 0 0")
    ET.SubElement(arm,"joint",name="kit_extension",type="slide",axis="1 0 0",range="0 .36",damping="4")
    box(arm,"kit_beam",(.102,.029,.024),(-.048,0,0),.13,".76 .54 .15 1")
    box(arm,"kit_palm",(.021,.057,.022),(.061,0,0),.065,".24 .31 .35 1")
    for name,sign in (("left",1),("right",-1)):
        finger = ET.SubElement(arm,"body",name=f"kit_{name}_finger",pos=f".09 {sign*.052} 0")
        ET.SubElement(finger,"joint",name=f"kit_{name}_jaw",type="slide",
                      axis=f"0 {-sign} 0",range="0 .035",damping="1")
        box(finger,f"kit_{name}_pad",(.040,.010,.022),(.023,0,0),.035,".20 .26 .28 1")
    # The powered rails and the two jaws are constrained mechanically. Exclude
    # their overlapping support surfaces so they cannot jam inside the housing.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root,"contact")
    for a,b in (("kit_mast","kit_carriage"),("kit_mast","kit_slide"),
                ("kit_carriage","kit_slide"),("kit_slide","kit_left_finger"),
                ("kit_slide","kit_right_finger")):
        ET.SubElement(contact,"exclude",body1=a,body2=b)
    actuators = root.find("actuator")
    for name,range_,kp,force in [
        ("kit_lift","0 .49","680","50"),("kit_extension","0 .36","450","28"),
        ("kit_left_jaw","0 .035","220","8"),("kit_right_jaw","0 .035","220","8"),
    ]:
        ET.SubElement(actuators,"position",name=name,joint=name,ctrlrange=range_,kp=kp,forcerange=f"-{force} {force}")
    for key in root.findall("./keyframe/key"):
        for attr in ("qpos","qvel","ctrl"):
            if key.get(attr):
                key.set(attr,key.get(attr)+" 0 0 0 0")
    return root


class LegActuation:
    """Feed only measured base and leg readings to a pinned walking actor."""

    def __init__(self, actor, observation_model, plant, leg_joints):
        self.actor = actor
        self.observation_model = observation_model
        self.observation = mujoco.MjData(observation_model)
        self.positions = np.array([plant.joint(name).qposadr[0] for name in leg_joints])
        self.velocities = np.array([plant.joint(name).dofadr[0] for name in leg_joints])
        self.actuators = []
        for name in leg_joints:
            joint = plant.joint(name).id
            actuators = np.flatnonzero((plant.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)
                                        & (plant.actuator_trnid[:,0]==joint))
            if len(actuators)!=1:
                raise ValueError(f"Expected one actuator for {name}")
            self.actuators.append(int(actuators[0]))

    def read_state(self, data):
        # This object only computes observations and receives policy torques.
        # The real plant advances independently through mj_step.
        self.observation.qpos[:7] = data.qpos[:7]
        self.observation.qvel[:6] = data.qvel[:6]
        self.observation.qpos[7:] = data.qpos[self.positions]
        self.observation.qvel[6:] = data.qvel[self.velocities]
        self.observation.time = data.time
        mujoco.mj_forward(self.observation_model,self.observation)

    def reset(self, data):
        self.read_state(data)
        self.actor.reset(self.observation_model,self.observation)

    def step(self, data, command):
        self.read_state(data)
        self.actor.step(self.observation_model,self.observation,command)
        data.ctrl[self.actuators] = self.observation.ctrl


def g1_arm(xml, leg_joints):
    """Use Menagerie's articulated G1 with dexterous hands and torque-driven legs."""
    if hashlib.sha256(Path(xml).read_bytes()).hexdigest()!=G1_HANDS_MJCF_SHA256:
        raise ValueError("Expected the pinned G1 Menagerie model with hands")
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir",str(Path(xml).parent/"assets"))
    actuators = root.find("actuator")
    for actuator in actuators:
        if actuator.get("joint") in leg_joints:
            actuator.tag = "motor"
            actuator.attrib.pop("class",None)
    wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    ET.SubElement(wrist,"site",name="reach_tip",pos=".171 -.003 .006")
    box(wrist,"reach_pad",(.022,.018,.016),(.151,-.003,.006),.012,".20 .24 .28 1")
    return root
