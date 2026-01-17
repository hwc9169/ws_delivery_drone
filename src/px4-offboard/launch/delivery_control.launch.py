#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_yaml = os.path.join(
        get_package_share_directory("px4_offboard"),
        "station_locations.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "station_yaml",
                default_value=default_yaml,
                description="Path to station_locations.yaml",
            ),
            DeclareLaunchArgument(
                "system_address",
                default_value="udp://:14540",
                description="MAVSDK system address",
            ),
            Node(
                package="px4_offboard",
                executable="delivery_control",
                name="delivery_control",
                output="screen",
                parameters=[
                    {"station_yaml": LaunchConfiguration("station_yaml")},
                    {"system_address": LaunchConfiguration("system_address")},
                ],
            ),
        ]
    )
