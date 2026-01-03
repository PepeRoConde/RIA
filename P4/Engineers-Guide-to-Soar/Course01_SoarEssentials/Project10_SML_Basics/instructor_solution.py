"""This script demonstrates a very simple SML interface for the Supplier Sorting agent.
It sets up a Soar agent, loads a Soar file, runs the agent, and reads agent output.
"""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import Course01_SoarEssentials.run_soar          # Importing this script sets up the env variables needed for us to import Python_sml_ClientInterface.

# Define the path for our Supplier Sorting Soar agent rules (in an OS-independent way).
AGENT_PATH = str(pathlib.Path(__file__).parent.joinpath("agent_code", "_firstload_suppliersort.soar").resolve())

#########################################################################


# STEP 1: Import the SML library. 
#   This is the only import needed to run Soar.
#   If this fails, your PYTHONPATH might not be configured properly to include the Soar bin directory. 
#     (Setting up your env paths should be handled already for you by importing the run_soar.py script above.)
import Python_sml_ClientInterface as sml


## Set up Soar

# STEP 2.1: Create a Soar Kernel. (A Kernel maintains and runs any number of Soar agents.)
# --> Use sml.Kernel.CreateKernelInNewThread(port) to create a Soar Kernel in a new thread that communicates using the given port.
# We'll tell the new Kernel to use port 12345. (You can use an arg of -1 to tell Soar to pick any available port.)
SOAR_PORT = 12345
print(f"Creating Soar Kernel using port {SOAR_PORT}")
kernel = sml.Kernel.CreateKernelInNewThread(SOAR_PORT)

# IMPORTANT: Always close any Kernel you create after you are done with it.
#   Scroll to the end of this file to STEP 2.2 and add code there to close this Kernel.

# STEP 3: Create a single Soar agent that will run inside this kernel, and give the agent an arbitrary name.
# (If you create additional agents in this kernel, they each need a unique name.)
# --> Use kernel.CreateAgent(agent_name) to have the invoked kernel create an agent with the given name.
agent = kernel.CreateAgent("My Agent")

# STEP 4.1: Now load production rule code into this agent.
# --> Use agent.LoadProductions(soar_file_path_string) to tell the invoked agent to load a .soar file.
print(f"Loading agent files from: {AGENT_PATH}")
agent.LoadProductions(AGENT_PATH)

# STEP 4.2: Test our load
# --> You can use agent.ExecuteCommandLine(cli_cmd) to run any CLI command and get its result as a string.
print(agent.ExecuteCommandLine("p propose*suppliersort-main*init"))


## Now set up input for this agent

# STEP 5: Get a pointer to the agent's input-link ID
# --> Use agent.GetInputLink() to get the agent's input-link ID.
il_id = agent.GetInputLink()

# STEP 6: Create a ^candidate-supplier structure on this ID.
# Create string ^name, float ^total-cost, and int ^total-sats WMEs underneath it.
# --> Use <ID>.CreateIdWME(attribute, value) to create an ID-type WME. (Notice the function name uses "Id" not "ID".)
# --> Use <ID>.CreateStringWME(attribute, value) to create a string-type WME.
# --> Use <ID>.CreateFloatWME(attribute, value) to create a float-type WME.
# --> Use <ID>.CreateIntWME(attribute, value) to create an int-type WME.
il_cand_chan_id = il_id.CreateIdWME("candidate-supplier")
il_cand_chan_id.CreateStringWME("name", "supplier01")
il_cand_chan_id.CreateFloatWME("total-cost", 24.99)
il_cand_chan_id.CreateIntWME("total-sats", 2)


## Now run the agent

# STEP 7: Run the agent on this input until the agent stops itself
# --> Use agent.RunSelfForever() to run the agent until something stops the agent.
print("Running Soar!")
agent.RunSelfForever()    # Runs the agent until it stops itself (an something else stopes it, like event listener code)

# STEP 9: Run the agent for 1 decision cycle more so that it sends the output that the agent created when it interrupted itself.
# --> Use agent.RunSelf(N) to run the agent for N decision cycles.
agent.RunSelf(1)          # Runs the agent for one more decision cycle so that the output is transmitted


## Now collect the agent's output

# STEP 8: Get a pointer to the agent's output-link ID.
# --> Use agent.GetOutputLink() to get a pointer to the agent's ^output-link ID.
# (GetOutputLink() returns None until the agent first sends output.)
ol_id = agent.GetOutputLink()
if ol_id is None:
    print("No output detected yet.")
else:
    try:
        # STEP 10: Read the size of the outputted list. (^supplier-list.count)
        # --> Use <ID>.FindByAttribute(attr_name, N) to get the Nth child WME with the given attribute name. (N=0 means the first WME.)
        # --> Use <WME>.ConvertToIdentifier() to cast a WME variable into an Identifier type, so you can access its children WMEs.
        # --> Use <WME>.ConvertToIntElement().GetValue() to get the integer value of an integer-type WME.
        list_count = (ol_id
            .FindByAttribute("supplier-list", 0).ConvertToIdentifier()
            .FindByAttribute("count", 0).ConvertToIntElement().GetValue())
        print(f"List size is {list_count}.")

        # STEP 11: If the count is > 0, read the name of the first supplier in the outputted list. (^supplier-list.first-supplier.name)
        # --> Use <WME>.GetValueAsString() To get the string value of a string-type WME. (WMEs are considered String type by default, so you don't need to convert the type.)
        if list_count > 0:
            first_chan_name = (ol_id
                .FindByAttribute("supplier-list", 0).ConvertToIdentifier()
                .FindByAttribute("first-supplier", 0).ConvertToIdentifier()
                .FindByAttribute("name", 0).GetValueAsString()
                )
            print(f"First supplier: {first_chan_name}")
    except AttributeError:
        # If FindByAttribute() doesn't find anything, it returns None.
        # So catch any error resulting from output missing an expected WME:
        print("ERROR IN OUTPUT HANDLER")

# STEP 2.2: Close the kernel before ending the script. Always a good practice!
# --> Use kernel.Shutdown() to close the kernel and deallocate any memory it or its agents are using.
print("\nGoodnight Soar.")
kernel.Shutdown()
