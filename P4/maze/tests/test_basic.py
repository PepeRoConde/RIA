#!/usr/bin/env python3
"""Basic test of maze simulation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.append(r"C:\Users\marce\Desktop\euu\cuarto\robotica\SoarSuite_9.6.4-Multiplatform\bin")

from enviroment.maze import Maze
from enviroment.agent_interface import MazeAgent

print("Creating 5x5 maze...")
maze = Maze(width=5, height=5, wall_probability=0.1)

print("\nInitial maze:")
maze.display(agent_pos=maze.start_pos, orientation='east')

print("\nCreating DFS agent...")
agent = MazeAgent(maze, strategy='dfs')

print("\nInitializing SOAR...")
if agent.initialize_soar():
    print("SOAR initialized successfully!")

    print("\nRunning 10 steps...")
    for step in range(10):
        print(f"\n--- Step {step + 1} ---", flush=True)
        print(f"Before run_step", flush=True)
        sys.stdout.flush()
        agent.run_step()
        print(f"After run_step", flush=True)
        sys.stdout.flush()

        if agent.has_reached_target():
            print("\nSUCCESS! Target reached!")
            break

    print("\nFinal state:")
    maze.display(agent_pos=(agent.x, agent.y),
                orientation=agent.ORIENTATION_NAMES[agent.orientation])

    stats = agent.get_stats()
    print(f"\nSteps: {stats['steps']}")
    print(f"Position: {stats['position']}")
    print(f"Orientation: {stats['orientation']}")

    agent.shutdown()
else:
    print("ERROR: Failed to initialize SOAR")
    sys.exit(1)

print("\nTest complete!")
