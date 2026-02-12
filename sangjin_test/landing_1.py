import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition

class Phase1FastApproach(Node):
    def __init__(self):
        super().__init__('phase1_fast_approach')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, qos_profile=qos)
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos_profile=qos)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)

        self.bridge = CvBridge()
        self.current_pos = [0.0, 0.0, 0.0]
        self.marker_world_pos = [None, None]
        self.cmd_setpoint = [None, None, None]
        self.target_alt = -2.0  # Phase 1 목표: 2m
        self.step_limit = 0.03  # 3cm 보폭 (빠름)

        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info('PHASE 1: 2m 고속 접근 노드 시작')

    def pos_callback(self, msg):
        self.current_pos = [msg.x, msg.y, msg.z]

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            corners, ids, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
            if ids is not None:
                m_center = np.mean(corners[0][0], axis=0)
                alt = abs(self.current_pos[2])
                m_per_px = (2 * alt * math.tan(1.047 / 2)) / frame.shape[1]
                dx = (m_center[1] - frame.shape[0]//2) * m_per_px
                dy = (m_center[0] - frame.shape[1]//2) * m_per_px
                self.marker_world_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy]
                if self.cmd_setpoint[0] is None: self.cmd_setpoint = list(self.current_pos)
        except: pass

    def timer_callback(self):
        self.publish_offboard_control_mode()
        if self.marker_world_pos[0] is not None:
            dist_xy = math.sqrt((self.marker_world_pos[0]-self.current_pos[0])**2 + (self.marker_world_pos[1]-self.current_pos[1])**2)
            
            # 위치 증분 계산
            self.cmd_setpoint[0] += np.clip(self.marker_world_pos[0] - self.cmd_setpoint[0], -self.step_limit, self.step_limit)
            self.cmd_setpoint[1] += np.clip(self.marker_world_pos[1] - self.cmd_setpoint[1], -self.step_limit, self.step_limit)
            
            # 20cm 이내 정렬 시 하강
            z_step = 0.01 if dist_xy < 0.20 else 0.0
            self.cmd_setpoint[2] += np.clip(self.target_alt - self.cmd_setpoint[2], -z_step, z_step)

            self.publish_setpoint(self.cmd_setpoint, float('nan'))

            # 완료 조건: 2m 도달 및 정렬 완료
            if abs(self.current_pos[2] - self.target_alt) < 0.1 and dist_xy < 0.1:
                self.get_logger().info('PHASE 1 완료! 노드를 종료합니다.')
                raise SystemExit # 노드 종료 (다음 노드 실행을 위해)

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
    node = Phase1FastApproach()
    try: rclpy.spin(node)
    except SystemExit: pass
    node.destroy_node(); rclpy.shutdown()

    
#### QGC에서 MPC_XY_P , MPC_Z_P 파라미터 0.5로 변경 (기체 심하게 기울어짐 방지)
####        MC_YAW_P 1로 변경