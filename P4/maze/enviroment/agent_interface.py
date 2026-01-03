import sys
from pathlib import Path
from typing import Set, Tuple

# Add Soar to Python path (adjust based on your Soar installation)
sys.path.append(str(Path(__file__).parent.parent.parent / "SoarSuite_9.6.2-Multiplatform" / "bin"))
import Python_sml_ClientInterface as sml

from .maze import Maze

class MazeAgent:
    """
    Python interface for Soar agent navigating maze.
    Manages agent state, sensory input, and discovered paths.
    Supports multiple navigation strategies.
    """
    
    # Orientation constants for DFS strategy
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    
    ORIENTATION_NAMES = {NORTH: 'north', EAST: 'east', SOUTH: 'south', WEST: 'west'}
    
    def __init__(self, maze: Maze, agent_name: str = "MazeNavigator", strategy: str = "naive"):
        self.maze = maze
        self.agent_name = agent_name
        self.strategy = strategy
        
        # Agent position
        self.x, self.y = maze.start_pos
        
        # Orientation (for DFS strategy) - initially facing EAST (right)
        self.orientation = self.EAST
        
        # Memory: discovered cells (x, y tuples)
        self.discovered: Set[Tuple[int, int]] = {(self.x, self.y)}
        
        # Movement history
        self.path = [(self.x, self.y)]
        
        # Soar components
        self.kernel = None
        self.agent = None
        self.input_link = None
        
        # Input link WMEs (Working Memory Elements)
        self.wmes = {}
        

    def initialize_soar(self):
        """Initialize Soar kernel and agent."""
        
        # Create Soar kernel
        try:
            self.kernel = sml.Kernel.CreateKernelInNewThread()
        except Exception as e:
            print(f"Error creating kernel: {e}")
            return False

        # Create agent
        try:
            self.agent = self.kernel.CreateAgent(self.agent_name)
        except Exception as e:
            print(f"Error creating agent: {e}")
            return False

        # Load agent rules based on strategy
        agent_path = Path(__file__).parent.parent / "agent"
        
        if self.strategy == "naive":
            rules_file = "maze-navigator-naive.soar"
        elif self.strategy == "dfs":
            rules_file = "maze-navigator-dfs.soar"
        else:
            print(f"Unknown strategy: {self.strategy}")
            return False
        
        result = self.agent.LoadProductions(str(agent_path / rules_file))

        if not result:
            print(f"Error loading productions: {self.agent.GetLastErrorDescription()}")
            return False

        # Get input link
        try:
            self.input_link = self.agent.GetInputLink()
        except Exception as e:
            print(f"Error getting input link: {e}")
            return False

        # Initialize input structures
        self._update_input_link()

        return True
 
    def _update_input_link(self):
        """Update Soar's input-link with current state."""
        # Clear old WMEs if they exist
        if self.wmes:
            for wme in self.wmes.values():
                wme.DestroyWME()
            self.wmes.clear()
        
        # Add position
        self.wmes['pos-x'] = self.input_link.CreateIntWME("pos-x", self.x)
        self.wmes['pos-y'] = self.input_link.CreateIntWME("pos-y", self.y)
        
        # Add target position
        target_x, target_y = self.maze.target_pos
        self.wmes['target-x'] = self.input_link.CreateIntWME("target-x", target_x)
        self.wmes['target-y'] = self.input_link.CreateIntWME("target-y", target_y)
        
        # Add sensors (relative to agent orientation for DFS)
        if self.strategy == "dfs":
            sensors = self._get_oriented_sensors()
            sensor_id = self.input_link.CreateIdWME("sensors")
            self.wmes['sensors'] = sensor_id
            self.wmes['sensor-front'] = sensor_id.CreateStringWME("front", "wall" if sensors['front'] else "clear")
            self.wmes['sensor-right'] = sensor_id.CreateStringWME("right", "wall" if sensors['right'] else "clear")
            self.wmes['sensor-back'] = sensor_id.CreateStringWME("back", "wall" if sensors['back'] else "clear")
            self.wmes['sensor-left'] = sensor_id.CreateStringWME("left", "wall" if sensors['left'] else "clear")
            
            # Add orientation
            self.wmes['orientation'] = self.input_link.CreateStringWME("orientation", 
                                                                        self.ORIENTATION_NAMES[self.orientation])
        else:
            # Absolute sensors for naive strategy
            sensors = self.maze.get_sensors(self.x, self.y)
            sensor_id = self.input_link.CreateIdWME("sensors")
            self.wmes['sensors'] = sensor_id
            self.wmes['wall-up'] = sensor_id.CreateStringWME("up", "wall" if sensors['up'] else "clear")
            self.wmes['wall-down'] = sensor_id.CreateStringWME("down", "wall" if sensors['down'] else "clear")
            self.wmes['wall-left'] = sensor_id.CreateStringWME("left", "wall" if sensors['left'] else "clear")
            self.wmes['wall-right'] = sensor_id.CreateStringWME("right", "wall" if sensors['right'] else "clear")
        
        # Add discovered cells count
        self.wmes['discovered-count'] = self.input_link.CreateIntWME("discovered-count", len(self.discovered))
        
        # Calculate distances to target
        dx = abs(target_x - self.x)
        dy = abs(target_y - self.y)
        self.wmes['distance-x'] = self.input_link.CreateIntWME("distance-x", dx)
        self.wmes['distance-y'] = self.input_link.CreateIntWME("distance-y", dy)
        self.wmes['manhattan-distance'] = self.input_link.CreateIntWME("manhattan-distance", dx + dy)
        
        # At target?
        at_target = (self.x, self.y) == self.maze.target_pos
        self.wmes['at-target'] = self.input_link.CreateStringWME("at-target", "yes" if at_target else "no")
    
    def _get_oriented_sensors(self) -> dict:
        """Get sensor readings relative to agent's current orientation."""
        # Get absolute sensors (already treats out-of-bounds as walls)
        abs_sensors = self.maze.get_sensors(self.x, self.y)
        
        # Map to relative directions based on orientation
        if self.orientation == self.NORTH:
            return {'front': abs_sensors['up'], 'right': abs_sensors['right'], 
                    'back': abs_sensors['down'], 'left': abs_sensors['left']}
        elif self.orientation == self.EAST:
            return {'front': abs_sensors['right'], 'right': abs_sensors['down'],
                    'back': abs_sensors['left'], 'left': abs_sensors['up']}
        elif self.orientation == self.SOUTH:
            return {'front': abs_sensors['down'], 'right': abs_sensors['left'],
                    'back': abs_sensors['up'], 'left': abs_sensors['right']}
        else:  # WEST
            return {'front': abs_sensors['left'], 'right': abs_sensors['up'],
                    'back': abs_sensors['right'], 'left': abs_sensors['down']}
    
    def process_output(self):
        """Process commands from Soar's output-link."""
        num_commands = self.agent.GetNumberCommands()
        
        for i in range(num_commands):
            command = self.agent.GetCommand(i)
            command_name = command.GetCommandName()
            
            if command_name == "move":
                direction = command.GetParameterValue("direction")
                success = self._execute_move(direction)
                
                # Mark command as complete
                if success:
                    command.AddStatusComplete()
                else:
                    command.AddStatusError()
            
            elif command_name == "turn":
                direction = command.GetParameterValue("direction")
                self._execute_turn(direction)
                command.AddStatusComplete()
    
    def _execute_move(self, direction: str) -> bool:
        """
        Execute a move command.
        
        Args:
            direction: 'forward', 'backward', or absolute ('up', 'down', 'left', 'right')
            
        Returns:
            True if move successful, False otherwise
        """
        if self.strategy == "dfs":
            # For DFS, interpret relative directions
            if direction == "forward":
                direction = self._get_absolute_direction(0)
            elif direction == "backward":
                direction = self._get_absolute_direction(2)
            # Otherwise assume it's already absolute
        
        new_x, new_y = self.x, self.y
        
        if direction == "up":
            new_y -= 1
        elif direction == "down":
            new_y += 1
        elif direction == "left":
            new_x -= 1
        elif direction == "right":
            new_x += 1
        else:
            print(f"Unknown direction: {direction}")
            return False
        
        # Check if move is valid
        if self.maze.is_wall(new_x, new_y):
            print(f"Cannot move {direction} - wall detected")
            return False
        
        # Execute move
        self.x, self.y = new_x, new_y
        self.path.append((self.x, self.y))
        self.discovered.add((self.x, self.y))
        
        print(f"Moved {direction} to ({self.x}, {self.y})")
        
        # Update input link with new state
        self._update_input_link()
        
        return True
    
    def _execute_turn(self, direction: str):
        """
        Execute a turn command (for DFS strategy).
        
        Args:
            direction: 'left' or 'right'
        """
        if direction == "left":
            self.orientation = (self.orientation - 1) % 4
        elif direction == "right":
            self.orientation = (self.orientation + 1) % 4
        
        print(f"Turned {direction}, now facing {self.ORIENTATION_NAMES[self.orientation]}")
        
        # Update input link with new orientation
        self._update_input_link()
    
    def _get_absolute_direction(self, relative_offset: int) -> str:
        """
        Convert relative direction to absolute.
        
        Args:
            relative_offset: 0=forward, 1=right, 2=back, 3=left
            
        Returns:
            Absolute direction string
        """
        absolute_orientation = (self.orientation + relative_offset) % 4
        
        if absolute_orientation == self.NORTH:
            return "up"
        elif absolute_orientation == self.EAST:
            return "right"
        elif absolute_orientation == self.SOUTH:
            return "down"
        else:  # WEST
            return "left"
    
    def run_step(self):
        """Run one decision cycle."""
        self.agent.RunSelf(1)
        self.process_output()
    
    def has_reached_target(self) -> bool:
        """Check if agent has reached the target."""
        return (self.x, self.y) == self.maze.target_pos
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            'position': (self.x, self.y),
            'target': self.maze.target_pos,
            'steps': len(self.path) - 1,
            'discovered': len(self.discovered),
            'total_cells': self.maze.width * self.maze.height,
            'orientation': self.ORIENTATION_NAMES[self.orientation] if self.strategy == "dfs" else None
        }
    
    def shutdown(self):
        """Clean shutdown of Soar."""
        if self.kernel:
            self.kernel.Shutdown()
            del self.kernel
