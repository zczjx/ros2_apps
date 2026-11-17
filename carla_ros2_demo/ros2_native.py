#!/usr/bin/env python

# Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

import argparse
import json
import logging
import math
import signal

import carla


def _make_truth_publisher(topic):
    """Optional: publish the hero ground truth (ROS convention) as nav_msgs/Odometry."""
    try:
        import rclpy
        from nav_msgs.msg import Odometry
    except ImportError:
        logging.warning("rclpy not found (source your ROS2 setup.bash): ground truth topic disabled")
        return None
    rclpy.init()
    node = rclpy.create_node("carla_ground_truth")
    pub = node.create_publisher(Odometry, topic, 10)

    def publish(vehicle, sim_time):
        tr = vehicle.get_transform()
        vel = vehicle.get_velocity()
        av = vehicle.get_angular_velocity()
        yaw = -math.radians(tr.rotation.yaw)  # CARLA left-handed -> ROS right-handed
        msg = Odometry()
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time % 1.0) * 1e9)
        msg.header.frame_id = "map"
        msg.child_frame_id = "hero"
        msg.pose.pose.position.x = tr.location.x
        msg.pose.pose.position.y = -tr.location.y
        msg.pose.pose.position.z = tr.location.z
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.twist.twist.linear.x = math.hypot(vel.x, vel.y)
        msg.twist.twist.angular.z = -math.radians(av.z)
        pub.publish(msg)

    return publish


def _setup_vehicle(world, config, spawn_point=0):
    logging.debug("Spawning vehicle: {}".format(config.get("type")))

    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name", config.get("id")) 

    return  world.spawn_actor(
        bp,
        map_.get_spawn_points()[spawn_point],
        attach_to=None)


def _setup_sensors(world, vehicle, sensors_config):
    bp_library = world.get_blueprint_library()

    sensors = []
    for sensor in sensors_config:
        logging.debug("Spawning sensor: {}".format(sensor))

        bp = bp_library.filter(sensor.get("type"))[0]
        bp.set_attribute("ros_name", sensor.get("id")) 
        bp.set_attribute("role_name", sensor.get("id")) 
        for key, value in sensor.get("attributes", {}).items():
            bp.set_attribute(str(key), str(value))

        wp = carla.Transform(
            location=carla.Location(x=sensor["spawn_point"]["x"], y=-sensor["spawn_point"]["y"], z=sensor["spawn_point"]["z"]),
            rotation=carla.Rotation(roll=sensor["spawn_point"]["roll"], pitch=-sensor["spawn_point"]["pitch"], yaw=-sensor["spawn_point"]["yaw"])
        )

        sensors.append(
            world.spawn_actor(
                bp,
                wp,
                attach_to=vehicle
            )
        )

        sensors[-1].enable_for_ros()

    return sensors


def main(args):
    # 让 kill/后台关闭也走 finally 清理车辆和传感器
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    world = None
    vehicle = None
    sensors = []
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        with open(args.file) as f:
            config = json.load(f)

        vehicle = _setup_vehicle(world, config, args.spawn_point)
        sensors = _setup_sensors(world, vehicle, config.get("sensors", []))

        _ = world.tick()

        vehicle.set_autopilot(True)

        truth_pub = _make_truth_publisher(args.truth_topic) if args.truth_topic else None

        logging.info("Running...")

        while True:
            _ = world.tick()
            if truth_pub:
                truth_pub(vehicle, world.get_snapshot().timestamp.elapsed_seconds)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        if original_settings:
            world.apply_settings(original_settings)

        for sensor in sensors:
            sensor.destroy()

        if vehicle:
            vehicle.destroy()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA ROS2 native')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', default='', required=True, help='File to be executed')
    argparser.add_argument('--spawn-point', default=0, type=int, help='index of the spawn point (default: 0)')
    argparser.add_argument('--truth-topic', default='/carla/hero/ground_truth', help="nav_msgs/Odometry ground truth topic, '' to disable")
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    main(args)
