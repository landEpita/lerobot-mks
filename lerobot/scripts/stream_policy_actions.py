"""
Minimal control script for running a policy on the robot **without** recording a dataset.

How it works
============
1. Builds the robot from a config (replace `robot_cfg` with yours).
2. Loads a pretrained ACT policy (point `pretrained_path` to your checkpoint).
3. Streams observations ➜ policy ➜ actions at the desired FPS.
4. Stops after `EPISODE_TIME_S` **or** automatically when all joints have been idle for a while.
"""

import time
from dataclasses import dataclass
from collections import deque

import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_utils import predict_action
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.robot_devices.motors.modbus_rtu_motor import ModbusRTUMotorsBus
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.policies.act.configuration_act import ACTConfig

########################################################################################
# USER‑EDITABLE SECTION                                                                #
########################################################################################
# 1. Provide your own `robot_cfg` (example below).
# 2. Point `pretrained_path` in `policy_cfg` to your checkpoint.
# 3. Adjust runtime parameters (FPS, EPISODE_TIME_S, etc.).
########################################################################################

# --- Example robot configuration -----------------------------------------------------
from lerobot.common.robot_devices.robots.configs import (
    FeetechMotorsBusConfig,
    MonRobot7AxesConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import ModbusRTUMotorsBusConfig
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.common.policies.act.configuration_act import NormalizationMode

robot_cfg = MonRobot7AxesConfig(
    leader_arms={
        "left": FeetechMotorsBusConfig(
            port="/dev/tty.usbmodem58FD0166391",
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                "gripper": [6, "sts3215"],
            },
            mock=False,
        )
    },
    follower_arms={
        "left": FeetechMotorsBusConfig(
            port="/dev/tty.usbmodem58FD0162261",
            motors={
                "shoulder_pan": [1, "sts3215"],
                "shoulder_lift": [2, "sts3215"],
                "elbow_flex": [3, "sts3215"],
                "wrist_flex": [4, "sts3215"],
                "wrist_roll": [5, "sts3215"],
                "gripper": [6, "sts3215"],
            },
            mock=False,
        ),
        "rail_lineaire": ModbusRTUMotorsBusConfig(
            port="/dev/tty.usbserial-BG00Q7CQ",
            motors={"axe_translation": (1, "NEMA17_MKS42D")},
            baudrate=115200,
        ),
    },
    cameras={
        "webcam": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
            color_mode="rgb",
            channels=3,
            rotation=None,
            mock=False,
        ),
        "camD": OpenCVCameraConfig(
            camera_index=1,
            fps=30,
            width=640,
            height=480,
            color_mode="rgb",
            channels=3,
            rotation=None,
            mock=False,
        ),
    },
    max_relative_target=None,
    gripper_open_degree=None,
    mock=False,
    calibration_dir=".cache/calibration/so100b",
)

policy_cfg = ACTConfig(
    n_obs_steps=1,
    normalization_mapping={
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    },
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.mounted": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        ),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    device="cuda",
    use_amp=False,
    chunk_size=100,
    n_action_steps=100,
    vision_backbone="resnet18",
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    dim_model=512,
    n_heads=8,
    dim_feedforward=3200,
    n_encoder_layers=4,
    n_decoder_layers=1,
    use_vae=True,
    latent_dim=32,
    dropout=0.1,
    kl_weight=10.0,
    optimizer_lr=1e-05,
    optimizer_weight_decay=0.0001,
    optimizer_lr_backbone=1e-05,
    pretrained_path="/Users/thomas/Documents/lbc/robot/lerobot/model/mks2",
)

########################################################################################
# RUNTIME PARAMETERS                                                                   #
########################################################################################
FPS = 30
EPISODE_TIME_S = 50  # seconds — set to None for infinite runtime

########################################################################################
# CONTROL LOOP                                                                         #
########################################################################################


@dataclass
class ControlParams:
    """Minimal set of parameters for action streaming."""

    fps: int = FPS
    episode_time_s: float | None = EPISODE_TIME_S


@safe_disconnect
def run_actions(robot: Robot, policy_cfg: ACTConfig, params: ControlParams) -> None:
    """Stream actions from a policy to the robot until timeout or auto‑stop."""

    # ‑‑‑ Connect robot ‑‑‑
    if not robot.is_connected:
        robot.connect()

    # Enable torque on follower arms if needed
    for arm in robot.follower_arms.values():
        if isinstance(arm, ModbusRTUMotorsBus):
            arm.write("Torque_Enable", 1)

    # Load policy once
    policy = make_policy(policy_cfg, ds_meta=None)
    device = get_safe_torch_device(policy_cfg.device)

    fifo: deque[torch.Tensor] = deque(maxlen=20)
    per_axis_thresh = torch.tensor([0.5, 0.5, 0.7, 0.7, 0.7, 0.1, 1500])

    start_episode_t = time.perf_counter()
    timestamp = 0.0

    print(">>> Running policy…  (Ctrl‑C to stop)")
    while params.episode_time_s is None or timestamp < params.episode_time_s:
        loop_start_t = time.perf_counter()

        # 1) Observation ↦ policy ↦ action
        obs = robot.capture_observation()
        act = predict_action(obs, policy, device, policy_cfg.use_amp)
        sent_act = robot.send_action(act)
        fifo.append(sent_act.clone())

        # 2) Auto‑stop if no movement
        if len(fifo) == fifo.maxlen:
            std_per_motor = torch.std(torch.stack(list(fifo)), dim=0)
            if torch.all(std_per_motor < per_axis_thresh):
                print("Auto‑stop: robot idle (std < threshold)")
                break

        # 3) Keep constant FPS
        if params.fps:
            busy_wait(max(0, 1 / params.fps - (time.perf_counter() - loop_start_t)))

        timestamp = time.perf_counter() - start_episode_t

    print(">>> Done!")


########################################################################################
# ENTRY POINT                                                                          #
########################################################################################
if __name__ == "__main__":
    params = ControlParams()
    robot = make_robot_from_config(robot_cfg)
    run_actions(robot, policy_cfg, params)
