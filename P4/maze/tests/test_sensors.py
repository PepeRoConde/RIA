#!/usr/bin/env python3
"""Test sensor readings"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from enviroment.maze import Maze

# Create a simple 10x10 maze
maze = Maze(width=10, height=10, wall_probability=0.2)

# Test sensors at position (9, 0) - right edge
pos = (9, 0)
sensors = maze.get_sensors(pos[0], pos[1])

print(f"Position: {pos}")
print(f"Sensors: {sensors}")
print(f"  Up (x={pos[0]}, y={pos[1]-1}): {'wall' if sensors['up'] else 'clear'}")
print(f"  Down (x={pos[0]}, y={pos[1]+1}): {'wall' if sensors['down'] else 'clear'}")
print(f"  Left (x={pos[0]-1}, y={pos[1]}): {'wall' if sensors['left'] else 'clear'}")
print(f"  Right (x={pos[0]+1}, y={pos[1]}): {'wall' if sensors['right'] else 'clear'}")

# Check if right edge is properly detected as wall
print(f"\nis_wall(10, 0): {maze.is_wall(10, 0)}")
print(f"is_valid_position(10, 0): {maze.is_valid_position(10, 0)}")
