import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition

class Phase2PrecisionLand(Node):
    def __init__(self):
        super().__init__('phase2_precision_land')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, qos_profile=qos)
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos_profile=qos)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)

        self.bridge = CvBridge()
        self.current_pos = [0.0, 0.0, 0.0]
        self.marker_world_pos = [None, None]
        self.cmd_setpoint = [None, None, None]
        self.target_yaw, self.cmd_yaw = 0.0, 0.0
        self.is_yaw_init = False

        self.target_alt = -0.3   # Phase 2 최종 목표: 0.3m
        self.step_limit = 0.003  # 3mm 초정밀 보폭
        self.yaw_step = 0.001    # 부드러운 회전 보폭

        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info('PHASE 2: 0.3m 초정밀 안착 노드 시작')

    def pos_callback(self, msg):
        self.current_pos = [msg.x, msg.y, msg.z]
        if not self.is_yaw_init: self.cmd_yaw = msg.heading

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            corners, ids, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
            if ids is not None:
                corners_np = corners[0][0]
                m_center = np.mean(corners_np, axis=0)
                alt = abs(self.current_pos[2])
                m_per_px = (2 * alt * math.tan(1.047 / 2)) / frame.shape[1]
                dx = (m_center[1] - frame.shape[0]//2) * m_per_px
                dy = (m_center[0] - frame.shape[1]//2) * m_per_px
                self.marker_world_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy]
                
                # 방향(Yaw) 계산
                vec = corners_np[1] - corners_np[0]
                self.target_yaw = math.atan2(vec[1], vec[0])

                if self.cmd_setpoint[0] is None: 
                    self.cmd_setpoint = list(self.current_pos)
                    self.is_yaw_init = True
        except: pass

    def timer_callback(self):
        self.publish_offboard_control_mode()
        if self.marker_world_pos[0] is not None:
            dist_xy = math.sqrt((self.marker_world_pos[0]-self.current_pos[0])**2 + (self.marker_world_pos[1]-self.current_pos[1])**2)
            
            # 초정밀 위치 업데이트
            self.cmd_setpoint[0] += np.clip(self.marker_world_pos[0] - self.cmd_setpoint[0], -self.step_limit, self.step_limit)
            self.cmd_setpoint[1] += np.clip(self.marker_world_pos[1] - self.cmd_setpoint[1], -self.step_limit, self.step_limit)
            
            # 5cm 이내 초정밀 정렬 시 0.3m까지 하강
            z_step = 0.003 if dist_xy < 0.05 else 0.0
            self.cmd_setpoint[2] += np.clip(self.target_alt - self.cmd_setpoint[2], -z_step, z_step)

            # 방향 정렬
            diff_yaw = self.target_yaw - self.cmd_yaw
            while diff_yaw > math.pi: diff_yaw -= 2*math.pi
            while diff_yaw < -math.pi: diff_yaw += 2*math.pi
            self.cmd_yaw += np.clip(diff_yaw, -self.yaw_step, self.yaw_step)

            self.publish_setpoint(self.cmd_setpoint, self.cmd_yaw)

            if abs(self.current_pos[2] - self.target_alt) < 0.05 and dist_xy < 0.03:
                self.get_logger().info('모든 정밀 착륙 과정 종료.')
                # 여기서 자동 Land 명령을 추가할 수 있습니다.

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position, msg.velocity = True, False
        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self, pos, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(pos[0]), float(pos[1]), float(pos[2])]
        msg.yaw = float(yaw)
        self.trajectory_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = Phase2PrecisionLand()
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

#### QGC에서 MPC_XY_P , MPC_Z_P 파라미터 0.5로 변경 (기체 심하게 기울어짐 방지)
####        MC_YAW_P 1로 변경