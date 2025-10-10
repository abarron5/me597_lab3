import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_sub')
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.callback,
            10
        )

    def callback(self, msg):
        print(f"AMCL pose: x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = TestSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
