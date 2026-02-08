import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# PX4 제어 메시지
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleStatus

class PrecisionLandingNode(Node):
    def __init__(self):
        super().__init__('precision_landing_node')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # 구독 및 발행 설정
        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, qos_profile)
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        self.bridge = CvBridge()
        
        # ArUco 설정 (버전 대응)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        # --- 제어 파라미터 수정 ---
        self.gain = 0.003          
        self.max_speed = 0.5       # 최대 속도 제한 (안전 장치)
        self.vx, self.vy, self.vz = 0.0, 0.0, 0.0

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('감도가 조절된 Precision Landing Node가 시작되었습니다.')

    def status_callback(self, msg):
        pass

    def timer_callback(self):
        # Offboard 하트비트
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        self.offboard_mode_pub.publish(offboard_msg)
        
        # 속도 명령 발행
        self.publish_trajectory(self.vx, self.vy, self.vz)

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2

            if self.use_new_api:
                corners, ids, rejected = self.detector.detectMarkers(frame)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

            if ids is None:
                # 마커를 놓치면 천천히 멈추도록 설정 (급정거 방지)
                self.vx *= 0.8
                self.vy *= 0.8
                self.vz = 0.0  # 하강 중지
            else:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                m_center = np.mean(corners[0][0], axis=0)
                m_x, m_y = int(m_center[0]), int(m_center[1])

                err_x = m_x - center_x
                err_y = m_y - center_y

                # 속도 계산 및 최대 속도 제한 (Clipping)
                raw_vx = float(-err_y * self.gain)
                raw_vy = float(err_x * self.gain)
                
                self.vx = np.clip(raw_vx, -self.max_speed, self.max_speed)
                self.vy = np.clip(raw_vy, -self.max_speed, self.max_speed)
                self.vz = 0.15  # 하강 속도도 조금 낮춤 (기존 0.2 -> 0.15)

                self.get_logger().info(f"마커 추적 중 - VX: {self.vx:.3f}, VY: {self.vy:.3f}")

            # 시각화
            cv2.line(frame, (center_x-10, center_y), (center_x+10, center_y), (255,0,0), 2)
            cv2.line(frame, (center_x, center_y-10), (center_x, center_y+10), (255,0,0), 2)
            cv2.imshow("Drone Landing View", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'에러: {e}')

    def publish_trajectory(self, vx, vy, vz):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [vx, vy, vz]
        msg.yaw = float('nan')
        self.trajectory_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()