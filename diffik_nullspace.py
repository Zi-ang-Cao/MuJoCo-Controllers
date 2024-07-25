"""
Differential IK controller for the Franka Panda robot arm. (7 joints)
In the xml file, the actuator are in 'position' type! This is important for the joint space controller to work.
The default class "panda" specifies the joint range to be range="-2.8973 2.8973"!

Usage: 
    mjpython diffik_nullspace.py  # Default robot is franka panda
    mjpython diffik_nullspace.py -n kinova
    mjpython diffik_nullspace.py -n ur5
    mjpython diffik_nullspace.py -n mobi_dex
    mjpython diffik_nullspace.py -n mobi_dex -d  # Enable debug mode

Author: @Zi-ang-Cao
Date: July 2024
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import click

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn_7dof = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
Kn_6dof = np.asarray([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
Kn = {
    6: Kn_6dof,
    7: Kn_7dof,
}

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


@click.command()
@click.option(
    "--debug", "-d", type=bool, is_flag=True, default=False, help="Enable debug mode."
)
@click.option(
    "--name_of_robot",
    "-n",
    type=click.Choice(["ur5", "ur5e", "franka", "panda", "kinova", "mobi_dex"]),
    default="franka",
    help="Name of the robot",
)
def main(debug, name_of_robot) -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    print(f"Using robot: {name_of_robot}")
    control_point_name = "attachment_site"
    dof = 6 if name_of_robot in ["ur5", "ur5e"] else 7
    partial_nullspace = False
    if "ur5" in name_of_robot:
        # Load the model and data.
        model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")

        body_names = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]

        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
    elif "franka" in name_of_robot or "panda" in name_of_robot:
        model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
        body_names = [
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
        ]

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
    elif "kinova" in name_of_robot:
        model = mujoco.MjModel.from_xml_path("kinova_gen3/scene.xml")
        body_names = [
            "shoulder_link",
            "half_arm_1_link",
            "half_arm_2_link",
            "forearm_link",
            "spherical_wrist_1_link",
            "spherical_wrist_2_link",
            "bracelet_link",
        ]

        joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
    elif "mobi_dex" in name_of_robot:
        partial_nullspace = True
        dof = 7
        model = mujoco.MjModel.from_xml_path("mobi_dex/scene.xml")

        joint_names = [
            # "joint_mobile_base_and_kinova_base",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            # "joint_eef_to_hand_palm",
        ]

    # Load the model and data.
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_id = model.site(control_point_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Damped least squares.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

            if partial_nullspace:
                # For Mobi_dex, we only put nullspace control for the 7 joints that belong to the Kinova arm
                dq += (eye - np.linalg.pinv(jac) @ jac)[:, :7] @ (
                    Kn[7] * (q0[:7] - data.qpos[dof_ids][:7])
                )
            else:
                dq += (eye - np.linalg.pinv(jac) @ jac) @ (
                    Kn[dof] * (q0 - data.qpos[dof_ids])
                )

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            np.clip(q, *model.jnt_range.T, out=q)

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)

            if debug:
                print(f"control freq: {1.0 / (time.time() - step_start)}")

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
