"""This script demonstrates how to use SML to remove old output and rerun the agent with new input."""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import Course01_SoarEssentials.run_soar          # Importing this script sets up the env variables needed for us to import Python_sml_ClientInterface.

# Define the path for our Supplier Sorting Soar agent rules (in an OS-independent way).
AGENT_PATH = str(pathlib.Path(__file__).parent.joinpath("agent_code", "_firstload_suppliersort.soar").resolve())

#########################################################################


import Python_sml_ClientInterface as sml


def read_first_supplier(soar_agent):
    # Collect output from Soar (GetOutputLink() returns None until the agent first sends output.)
    ol_id = soar_agent.GetOutputLink()
    if ol_id is None:
        print("No output detected yet.")
    else:
        try:
            # Use "FindByAttribute" to get the Nth WME with the given attribute name. (0 means we want the first WME.)
            list_count = (ol_id
                .FindByAttribute("supplier-list", 0).ConvertToIdentifier()
                .FindByAttribute("count", 0).ConvertToIntElement().GetValue())
            print(f"List size is {list_count}.")

            if list_count > 0:
                first_chan_name = (ol_id
                    .FindByAttribute("supplier-list", 0).ConvertToIdentifier()
                    .FindByAttribute("first-supplier", 0).ConvertToIdentifier()
                    .FindByAttribute("name", 0).GetValueAsString()
                    )
                print(f"First supplier: {first_chan_name}")
        except AttributeError:
            # Catch any error resulting from output missing an expected WME:
            print("ERROR IN OUTPUT HANDLER")


## Set up Soar

# Create a Soar Kernel to be accessible on port 12345.
SOAR_PORT = 12345
print(f"Creating Soar Kernel using port {SOAR_PORT}")
kernel = sml.Kernel.CreateKernelInNewThread(SOAR_PORT)

# Create a single Soar agent with an arbitrary name.
agent = kernel.CreateAgent("My Agent")

# Now load production rule code into this agent.
print(f"Loading agent files from: {AGENT_PATH}")
agent.LoadProductions(AGENT_PATH)


## Now set up input for this agent

il_id = agent.GetInputLink()

# Create a ^candidate-supplier structure on this id.
il_cand_chan_id = il_id.CreateIdWME("candidate-supplier")
il_cand_chan_id.CreateStringWME("name", "supplier01")
il_cand_chan_id.CreateFloatWME("total-cost", 24.99)
il_cand_chan_id.CreateIntWME("total-sats", 2)


## Now run the agent

# STEP 1: Open up a Java Debugger for this agent
# --> Use agent.SpawnDebugger() to open a debugger for the specified agent.
agent.SpawnDebugger()

# Wait for a keystroke to continue running this script.
# (The debugger will close when the script ends, so without this, you wouldn't see the debugger open.)
input("*** Press Enter to run Soar ***")

print("Running Soar!")
agent.RunSelfForever()    # Runs the agent until it stops itself (an something else stopes it, like event listener code)
agent.RunSelf(1)          # Runs the agent for one more decision cycle so that the output is transmitted


## Now read (and print) the agent's output

read_first_supplier(agent)


## Now remove old input and add new input

# STEP 2: Remove the existing candidate-supplier input from the agent's input-link.
# --> You can use <WME>.DestroyWME() to remove a single WME
# --> You can use <ID>.DestroyWME() to remove all WME children that are under an ID.
il_cand_chan_id.DestroyWME()

# IMPORTANT: You should always set any local Python WME/ID variables to None after using DestroyWME to destroy what they point to!
# After the WME is removed, we can no longer use it.
# Dereferencing a removed WME will cause a Segmentation Fault.
# (The SML variables and functions here in Python are mirrored to C++ artifacts under the hood.)
# Setting them to None lets your Python code gracefully catch attempts to reference removed WMEs.

# STEP 3: Set variables to None if they point to a WME or ID that you just destroyed.
il_cand_chan_id = None

# STEP 4: Try creating a string WME on the candidate-supplier ID we just destroyed.
# First, run this script and see this code catch the error.
# Then comment out your code from STEP 3 and run this again. Observe the seg fault.
# Restore your STEP 3 code and move on to STEP 5.
try:
    il_cand_chan_id.CreateStringWME("test", "not-destroyed")
except AttributeError as e:
    print("ERROR: Trying to reference a removed WME or ID!")
except Exception:
    print("This message will not be printed. Seg Faults are like that.")


# STEP 5.1: Uncomment this code to replace the None value of your removed input with new candidate-supplier input.
# (Notice that the supplier name here is different than before.)
il_cand_chan_id = il_id.CreateIdWME("candidate-supplier")
il_cand_chan_id.CreateStringWME("name", "supplier02")
il_cand_chan_id.CreateFloatWME("total-cost", 14.99)
il_cand_chan_id.CreateIntWME("total-sats", 2)

# Wait for a keystroke to continue running this script.
input("*** Press Enter to run Soar a second time ***")

# STEP 5.2: Uncomment this code to run the agent on this new input:
print("Running Soar!")
agent.RunSelfForever()    # Runs the agent until it stops itself (an something else stopes it, like event listener code)
agent.RunSelf(1)          # Runs the agent for one more decision cycle so that the output is transmitted

read_first_supplier(agent)


## Close the kernel before ending the script.

# Wait for a keystroke to continue running this script.
input("*** Press Enter to finish and close the debugger ***")

# Shut down
print("\nGoodnight Soar.")
kernel.Shutdown()
