"""
Test script to verify DestroyWME fix works correctly.
Tests the agent can properly create and destroy WMEs without errors.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from enviroment.maze import Maze
from enviroment.agent_interface import MazeAgent


def test_destroy_wme():
    """Test that WME creation and destruction works properly."""
    print("=" * 60)
    print("Testing DestroyWME Fix")
    print("=" * 60)

    # Create a small maze
    print("\n1. Creating maze...")
    maze = Maze(width=5, height=5, wall_probability=0.2)
    print("   Maze created successfully!")

    # Initialize agent
    print("\n2. Initializing Soar agent...")
    agent = MazeAgent(maze, strategy="dfs")

    if not agent.initialize_soar():
        print("   ERROR: Failed to initialize Soar agent!")
        return False

    print("   Soar agent initialized successfully!")

    # Test multiple update cycles to ensure WMEs are properly destroyed/created
    print("\n3. Testing WME updates (create/destroy cycles)...")

    for i in range(5):
        print(f"   Cycle {i + 1}...")

        # Run a step
        try:
            agent.run_step()
            print(f"      Step executed successfully")
        except Exception as e:
            print(f"      ERROR: {e}")
            return False

        # Manually trigger update to test destroy/create
        try:
            agent._update_input_link()
            print(f"      Input link updated successfully")
        except Exception as e:
            print(f"      ERROR during input link update: {e}")
            return False

    print("\n4. Testing output command processing...")

    # Run a few more steps to process commands
    for i in range(3):
        try:
            agent.run_step()
            print(f"   Command cycle {i + 1} processed successfully")
        except Exception as e:
            print(f"   ERROR during command processing: {e}")
            return False

    print("\n5. Cleaning up...")
    agent.shutdown()
    print("   Shutdown successful!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    # Display final stats
    stats = agent.get_stats()
    print(f"\nFinal Stats:")
    print(f"  Position: {stats['position']}")
    print(f"  Steps: {stats['steps']}")
    print(f"  Discovered: {stats['discovered']} cells")

    return True


if __name__ == "__main__":
    success = test_destroy_wme()
    exit(0 if success else 1)
