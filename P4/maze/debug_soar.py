#!/usr/bin/env python3
"""Debug Soar WME structure"""
import sys
from pathlib import Path

sys.path.append(r"C:\Users\marce\Desktop\euu\cuarto\robotica\SoarSuite_9.6.4-Multiplatform\bin")
import Python_sml_ClientInterface as sml

from enviroment.maze import Maze
from enviroment.agent_interface import MazeAgent

# Create simple maze
maze = Maze(width=5, height=5, wall_probability=0.1)
agent = MazeAgent(maze)

if not agent.initialize_soar():
    print("Failed to initialize")
    sys.exit(1)

# Move agent to edge manually
agent.x, agent.y = 4, 0  # Right edge
agent._update_input_link()

print(f"\nAgent at edge position: ({agent.x}, {agent.y})")
sensors = maze.get_sensors(agent.x, agent.y)
print(f"Python sensors: {sensors}")

# Print input-link BEFORE running
print("\n--- Soar Input-Link (BEFORE decision cycle) ---")
result = agent.agent.ExecuteCommandLine("print --depth 4 i2")
print(result)

# Run several cycles to get past initialization
print("\n--- Running 5 decision cycles ---")
for i in range(5):
    result = agent.agent.ExecuteCommandLine("step 1")
    print(f"Cycle {i+1}: {result.strip()}")

# Check what operators were proposed NOW
print("\n--- Checking Current Operator Proposals ---")
result = agent.agent.ExecuteCommandLine("print s1 ^operator")
print(result[:800])

# Check output-link for commands
print("\n--- Output-Link Commands ---")
result = agent.agent.ExecuteCommandLine("print --depth 5 i3")
print(result)

# Try to see the commands in detail
print("\n--- Command Details ---")
result = agent.agent.ExecuteCommandLine("print c3")
print(result)
result = agent.agent.ExecuteCommandLine("print c4")
print(result)

agent.shutdown()
