"""This script provides a quick way for you to run the Soar agents that you create in this course.
It also serves as a simple demo for how to run Soar from Python.
"""
from os import environ as env, getenv, system, name as os_name
import os
from pathlib import Path as Path
import sys
import inspect

# Get paths to the Soar Setup, Debugger, and executable launch scripts
if os_name == "posix":
    script_ext = ".sh"
    path_sep = ":"
else:
    script_ext = ".bat"
    path_sep = ";"

# First, we'll set the machine environment variables to point to the Soar distribution that is included in this course.
#   (You can comment this out if you already have these set up on your machine.)
#   Set SOAR_HOME, DYLD_LIBRARY_PATH, PATH, and PYTHONPATH environment variables
soar_run_abs_path = Path(__file__).parent.parent.joinpath("SoarSuite_9.6.2-Multiplatform").resolve()
soar_home_abs_path_str = str(Path(soar_run_abs_path).joinpath("bin"))
env["SOAR_HOME"] = soar_home_abs_path_str
env["DYLD_LIBRARY_PATH"] = soar_home_abs_path_str + path_sep + getenv("DYLD_LIBRARY_PATH","")
env["PATH"] = soar_home_abs_path_str + path_sep + getenv("PATH","")
env["PYTHONPATH"] = soar_home_abs_path_str + path_sep + getenv("PYTHONPATH","")

SOAR_SETUP_PATH = str(soar_run_abs_path.joinpath("setup"+script_ext))
SOAR_JDEBUGGER_PATH = str(soar_run_abs_path.joinpath("SoarJavaDebugger"+script_ext))
SOAR_CLI_PATH = str(soar_run_abs_path.joinpath("SoarCLI"+script_ext))

# Import the SML library. This is the only import needed to run Soar.
# If this fails, your PYTHONPATH is not configured properly to include the Soar bin folder,
#  or the setup script was not yet run to establish the Python library at that location.
sys.path.insert(1, soar_home_abs_path_str)
try:
    import Python_sml_ClientInterface as sml
except ImportError:
    system(SOAR_SETUP_PATH)
    import Python_sml_ClientInterface as sml

def run_soar_python(agent_files=None, agent_name:str="TestAgent"):
    """Run the given Soar agent in a new SoarJavaDebugger window, controlled from Python SML.
    The platform-specific files must first be copied to the bin folder.
    (Running run_soar_cli() or run_soar_debugger() will automatically perform the appropriate copy.)

    agent_files=None:
        The path(s) to the agent file you wish to load. (Can be a single string, or a list of strings.)
        Starts an empty agent if None provided.
    agent_name="TestAgent":
        The name to give the new agent. 
        (This is only used for differentiating multiple agents running in the same Kernel.)
    """
    # Create a Soar Kernel (a Kernel spawns and runs any number of Soar agents in its thread)
    # We tell the new Kernel to be accessible on port 12121. (Use -1 to tell Soar to pick any available port.)
    kernel = sml.Kernel.CreateKernelInNewThread(12121)

    # Create a single Soar agent that will run inside this kernel, and give it a name.
    # (If you create additional agents in this kernel, they each need a unique name.)
    agent = kernel.CreateAgent(agent_name)

    # Now load agent code into this agent
    if agent_files is not None:
        # Load the given agent code
        if isinstance(agent_files, list):
            for file in agent_files:
                agent.LoadProductions(str(file))
        else:
            agent.LoadProductions(str(agent_files))
        
    # Open a debugger window so we can view the agent's processing
    agent.SpawnDebugger()

    ## Uncomment to run the agent from here:
    # agent.RunSelfForever()    # Runs the agent until an event listener stops it or it stops itself
    # agent.RunSelf(10)         # Runs the agent for the given number of decision cycles (steps)

    # Wait for a keystroke to finish the script.
    # (The debugger will close when the script ends, so without this, you wouldn't see it open.)
    input("Press Enter to shut down Soar...")

    # Close the kernel before ending the script. Always a good practice!
    kernel.Shutdown()

def run_soar_cli(agent_file:str=None):
    """Run the Soar executable from the terminal.
    
    agent_file=None:
        The path to the agent file you wish to load.
        Starts an empty agent if None provided.
    """
    if agent_file is None:
        system(SOAR_CLI_PATH)
    else:
        system(SOAR_CLI_PATH+" -s "+str(agent_file))

def run_soar_debugger(agent_file:str=None):
    """Run Soar with a new SoarJavaDebugger window.
    
    agent_file=None:
        The path to the agent file you wish to load.
        Starts an empty agent if None provided.
    """
    if agent_file is None:
        system(SOAR_JDEBUGGER_PATH+" run")
    else:
        target_file = agent_file
        if agent_file[0] == ".":
            # If the path is relative, we'll assume it's relative to the calling file's location
            calling_filename = inspect.stack()[1].filename 
            target_file = Path(calling_filename).parent.joinpath(agent_file).resolve()
        else:
            target_file = agent_file
        
        system(SOAR_JDEBUGGER_PATH+" -source "+str(target_file)+" run")
