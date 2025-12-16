import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
import numpy as np
from attrdict import AttrDict
import random

class RobotEnv():
    def __init__(self, renders, urdf):
        self.renders = renders
        self.urdf_file = urdf
        if self.renders:
            # 服务端打开图形GUI做渲染，需要独显，性能消耗大
            p.connect(p.GUI)
        else:
            # 不打开图形渲染，性能消耗小
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")  # 地面
        self.robot_id = p.loadURDF(self.urdf_file, useFixedBase=True)  # 机器人
        p.setRealTimeSimulation(False)
        p.setGravity(0, 0, -9.81)
        # 转变视角
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[1.0, 0, 0])
        # 机器人起始位姿设定
        robot_start_pos = [0, 0, 0]
        # RX,RY,RZ
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.robot_id, robot_start_pos, robot_start_orientation)
        # 关节信息
        self.joint_info = namedtuple("jointInfo",
                                     ["joint_id", "joint_name", "joint_lower_limit", "joint_upper_limit", "max_force",
                                      "axis_name", "joint_pos", "joint_rot", "parent_index"])
        # 可以控制的关节名称
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                               "wrist_2_joint", "wrist_3_joint"]
        # 关节类型
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joints = AttrDict()
        self.obs_id = []
        self.joint_soft_limit=((-360,360),(-360,360),(-360,360),(-360,360),(-360,360),(-360,360))
        self.current_joint=np.asarray([0,0,0,0,0,0])

    def generate_random_value(self,start, end, is_integer=True):
        """
        生成指定范围的随机值。
        :param start: 数值范围的起始值（包括）
        :param end: 数值范围的结束值（包括）
        :param is_integer: 是否生成整数，默认为 True
        :return: 随机数（整数或浮点数）
        """
        if is_integer:
            return random.randint(start, end)  # 生成整数
        else:
            return random.uniform(start, end)  # 生成浮点数
    def reset(self):
        for i in range(len(self.current_joint)):
            self.current_joint[i]=self.generate_random_value(self.joint_soft_limit[i][0],self.joint_soft_limit[i][1])
        self.set_joint_angles(self.current_joint)
        return self.current_joint
    def step(self,action):
        """
        与环境交互并返回新的状态、奖励、是否完成
        Args:
            action: 动作 一组关节增量

        Returns:

        """
        reward=0
        self.current_joint+=action
        self.set_joint_angles(self.current_joint)
        collide_status=self.is_collision()
        if collide_status:
            reward-=10
            return self.current_joint, collide_status
        return self.current_joint,collide_status

    def get_joint_num(self):
        num_joints = p.getNumJoints(self.robot_id)
        print("关节数量:", num_joints)
        # 打印每个关节的信息
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            max_force = info[10]
            axis_name = info[12].decode("utf-8")
            joint_pos = info[14]
            joint_rot = info[15]
            parent_index = info[16]
            info = self.joint_info(joint_id, joint_name, joint_lower_limit, joint_upper_limit, max_force, axis_name,
                                   joint_pos, joint_rot, parent_index)
            self.joints[info.joint_name] = info

    # def ik(self, robot_id, end_effector_index, target_position, target_quat, last_joints):
    #     last_joints = np.asarray(last_joints) / 180 * np.pi
    #     lower_limits = [-math.pi] * 6
    #     upper_limits = [math.pi] * 6
    #     joint_ranges = [2 * math.pi] * 6
    #     # 计算逆运动学
    #     ik_solution = p.calculateInverseKinematics(robot_id, end_effector_index, target_position,
    #                                                target_quat)
    #
    #     return np.asarray(ik_solution) * 180 / np.pi

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []
        joint_angles = joint_angles / 180 * np.pi
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.joint_id)
            forces.append(joint.max_force)
        p.setJointMotorControlArray(
            self.robot_id, indexes,
            p.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces
        )
        p.stepSimulation()
        time.sleep(1 / 240)

    def create_box(self, size, box_position, box_orientation):
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)  # 创建碰撞箱模型
        box_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=(0.1, 0.5, 0.1, 1))  # 创建视觉模型
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_id,
                          baseVisualShapeIndex=box_visual_id, basePosition=box_position,
                          baseOrientation=box_orientation)
        self.obs_id.append(box_id)

    def is_collision(self):
        for ids in self.obs_id:
            if bool(p.getContactPoints(bodyA=self.robot_id, bodyB=ids)):
                return True
        return False
