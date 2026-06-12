import rclpy
from rclpy.node import Node
from std_msgs.msg import String 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
import datetime 


class CollectCmdSubscriber(Node):
    def __init__(self):
        # Initialize node (name suggested with "subscriber" to avoid conflict with sender node)
        super().__init__("collect_cmd_subscriber")
        
        # -------------------------- Key: QoS configuration must match sender --------------------------
        # Must match Ros2CollectorClient's QoS (reliability + depth must be the same)
        # If sender is created with reliable=False, change to ReliabilityPolicy.BEST_EFFORT here
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Keep consistent with sender's depth=1
        )

        # Create subscriber: subscribe to /collect_cmd topic and bind callback function
        self.subscription = self.create_subscription(
            msg_type=String,
            topic="/collect_cmd",  # Topic name exactly matching sender
            callback=self.cmd_callback,
            qos_profile=self.qos_profile
        )
        self.subscription  # Prevent IDE warning about unused variable

        # Node startup log (log output to confirm startup status)
        self.get_logger().info("=" * 50)
        self.get_logger().info("Collect Command Subscriber Started!")
        self.get_logger().info(f"Topic to listen: /collect_cmd")
        self.get_logger().info(f"QoS Config: Reliable, Depth=1")
        self.get_logger().info("Waiting for 'start'/'stop' commands...")
        self.get_logger().info("=" * 50)

    def cmd_callback(self, msg):
        """
        Message callback function: triggered when command is received, prints detailed info
        :param msg: Received message (std_msgs/String), msg.data contains command content
        """
        # Get current time (accurate to milliseconds for sequence identification)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Validate command legality (only care about start/stop, filter invalid commands)
        if msg.data in ["start", "stop"]:
            self.get_logger().info(f"[{current_time}] 🔔 Received Valid Command: {msg.data.upper()}")
        else:
            self.get_logger().warn(f"[{current_time}] ⚠️ Received Invalid Command: '{msg.data}' (only 'start'/'stop' allowed)")


def main(args=None):
    # Initialize rclpy context
    rclpy.init(args=args)
    subscriber_node = CollectCmdSubscriber()
    
    try:
        rclpy.spin(subscriber_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\n[collect_cmd_subscriber] stopping...")
    finally:
        if rclpy.ok():
            subscriber_node.get_logger().info("Subscriber stopping...")
        subscriber_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()