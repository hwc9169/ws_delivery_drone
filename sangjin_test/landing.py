import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition

class OptimalLandingNode(Node):
    def __init__(self):
        super().__init__('optimal_landing_node')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, qos_profile)
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos_profile)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        self.bridge = CvBridge()
        self.current_pos = [0.0, 0.0, 0.0]
        self.target_pos = [None, None, None] # 실시간 업데이트될 목표 절대 좌표
        
        self.state = "SEARCHING"
        self.fov = 1.047
        self.kp = 0.5 # 위치 오차를 속도로 변환하는 게인 (상황에 따라 조절)

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        self.timer = self.create_timer(0.02, self.timer_callback) # 50Hz
        self.get_logger().info('최적화된 실시간 추적형 착륙 알고리즘 시작')

    def pos_callback(self, msg):
        self.current_pos = [msg.x, msg.y, msg.z]

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            if self.use_new_api:
                corners, ids, _ = self.detector.detectMarkers(frame)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                # 마커가 보일 때마다 목표 지점을 현재 드론 위치 기준으로 업데이트 (Update 단계)
                m_center = np.mean(corners[0][0], axis=0)
                err_px_x = m_center[0] - frame.shape[1] // 2
                err_px_y = m_center[1] - frame.shape[0] // 2

                alt = abs(self.current_pos[2])
                m_per_px = (2 * alt * math.tan(self.fov / 2)) / frame.shape[1]
                
                # 드론의 현재 절대 좌표에 픽셀 오차만큼 더해서 '마커의 절대 좌표'를 계산
                dx_body = -err_px_y * m_per_px
                dy_body = err_px_x * m_per_px
                
                # 마커가 있는 실제 지상의 절대 위치를 타겟으로 고정
                self.target_pos = [
                    self.current_pos[0] + dx_body,
                    self.current_pos[1] + dy_body,
                    -1.0 # 최종 목표 고도 1m
                ]
                
                if self.state == "SEARCHING":
                    self.state = "DESCENDING"
                    self.get_logger().info('마커 포착: 추적 및 하강 시작')

            cv2.imshow("Optimal View", frame)
            cv2.waitKey(1)
        except Exception: pass

    def timer_callback(self):
        # 오프보드 신호
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = True
        self.offboard_mode_pub.publish(offboard_msg)

        if self.state == "DESCENDING" and self.target_pos[0] is not None:
            # P 제어: (목표 위치 - 현재 위치) * 게인 = 필요 속도
            err_x = self.target_pos[0] - self.current_pos[0]
            err_y = self.target_pos[1] - self.current_pos[1]
            err_z = self.target_pos[2] - self.current_pos[2]

            # 속도 제한 (최대 0.5m/s로 천천히 안전하게)
            cmd_vel = [
                np.clip(err_x * self.kp, -0.5, 0.5),
                np.clip(err_y * self.kp, -0.5, 0.5),
                np.clip(err_z * self.kp, -0.3, 0.3)
            ]

            self.publish_velocity(cmd_vel)

            # 1m 고도 및 오차 10cm 이내면 정지
            if abs(err_z) < 0.1 and abs(err_x) < 0.1 and abs(err_y) < 0.1:
                self.state = "FINISHED"
                self.get_logger().info('최적 지점 정착 완료')

    def publish_velocity(self, vel):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.velocity = [float(vel[0]), float(vel[1]), float(vel[2])]
        self.trajectory_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OptimalLandingNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()