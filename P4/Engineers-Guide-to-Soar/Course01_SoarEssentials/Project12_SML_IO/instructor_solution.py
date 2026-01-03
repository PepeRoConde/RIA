"""This script demonstrates a more object-oriented way of handling Soar I/O."""
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import Course01_SoarEssentials.run_soar          # Importing this script sets up the env variables needed for us to import Python_sml_ClientInterface.

# Define the path for our Supplier Sorting Soar agent rules (in an OS-independent way).
AGENT_PATH = str(pathlib.Path(__file__).parent.joinpath("agent_code", "_firstload_suppliersort.soar").resolve())

#########################################################################


# Import the SML library
import Python_sml_ClientInterface as sml


class MyInputSupplier:
    """A class for defining an input candidate-supplier and adding it to an agent input-link"""
    def __init__(self, name) -> None:
        # Initialize the fields for augmentations of the candidate-supplier object
        self.name = name
        self.total_cost = 0.0
        self.total_score = 0
        self.total_sats = 0
        self.sustainability = 0
        self.availability = 0
        self.quality = 0
        self.packaging = 0
        self.speed = 0

        # Initialize the field for the WME of this candidate-supplier ID
        self.il_cand_chan_id = None

    def set_totals(self, total_cost, total_score, total_sats):
        # STEP 1: Set the values for this supplier's totals
        self.total_cost = total_cost
        self.total_score = total_score
        self.total_sats = total_sats

    def set_ranks(self, sustainability, availability, quality, packaging, speed):
        # STEP 2: Set the values for this supplier's ranks
        self.sustainability = sustainability
        self.availability = availability
        self.quality = quality
        self.packaging = packaging
        self.speed = speed

    def remove_from_input_link(self):
        """Destroy all input WMEs for this supplier"""
        # STEP 3: If the candidate-supplier ID has been added to the input-link, destroy it.
        #   (Recall: When the root WME is destroyed, all orphaned child WMEs are also destroyed.)
        if self.il_cand_chan_id != None:
            self.il_cand_chan_id.DestroyWME()
            self.il_cand_chan_id = None

    def push_to_input_link(self, input_link_id):
        # STEP 4.1: Call remove_from_input_link first so that we remove old input (if any) before adding new input
        self.remove_from_input_link()

        # STEP 4.2: Add this instance's data to the input-link as a new input candidate-supplier object
        self.il_cand_chan_id = input_link_id.CreateIdWME("candidate-supplier")
        self.il_cand_chan_id.CreateStringWME("name", self.name)
        self.il_cand_chan_id.CreateFloatWME("total-cost", self.total_cost)
        self.il_cand_chan_id.CreateIntWME("total-score", self.total_score)
        self.il_cand_chan_id.CreateIntWME("total-sats", self.total_sats)
        self.il_cand_chan_id.CreateIntWME("sustainability", self.sustainability)
        self.il_cand_chan_id.CreateIntWME("availability", self.availability)
        self.il_cand_chan_id.CreateIntWME("quality", self.quality)
        self.il_cand_chan_id.CreateIntWME("packaging", self.packaging)
        self.il_cand_chan_id.CreateIntWME("speed", self.speed)


def read_ol_supplier_list(ol_supplier_id):
    return_list = []
    try:
        # Recursively follow "^next" links in the returned list to read the recommended list of suppliers
        # STEP 5: Loop over the child WMEs of the given supplier-list ID
        # --> Use <ID>.GetNumberChildren() to get the count of that ID's child WMEs
        for i in range(ol_supplier_id.GetNumberChildren()):
            # Loop through all supplier augmentations and process each in a case-based manner
            ## (TODO: Explain in slides why looping is more efficient than FindByAttribute for each attribute)

            # STEP 6: Read the i-th child WME of the supplier-list ID and get its attribute name
            # --> Use <ID>.GetChild(i) to get the i-th child WME of that ID
            # --> Use <ID>.GetAttribute() to get the attribute name of that WME
            ol_list_item_wme = ol_supplier_id.GetChild(i)
            attr = ol_list_item_wme.GetAttribute()

            # Process the different supplier attributes in a case-based manner
            if attr == "name":
                return_list.insert(0,ol_list_item_wme.GetValueAsString())
            elif attr == "next":
                # Get the value of this WME as an ID
                ol_next_item_id = ol_list_item_wme.ConvertToIdentifier()
                # Recursively append the list elements downstream from this one
                return_list.extend(read_ol_supplier_list(ol_next_item_id))
    except AttributeError as e:
        print("ERROR reading a supplier!")
        print(repr(e))
    
    # Return the list of this item and all its descendent items
    return return_list


def get_supplier_list(soar_agent):
    # Collect output from Soar
    ol_id = soar_agent.GetOutputLink()
    if ol_id is None:
        print("No output detected yet.")
        return []
    
    # Check the size of the output list
    try:
        ol_list_id = ol_id.FindByAttribute("supplier-list", 0).ConvertToIdentifier()
        list_size = ol_list_id.FindByAttribute("count", 0).ConvertToIntElement().GetValue()
    except AttributeError:
        print("ERROR: No supplier-list in agent output!")
        return []
    
    # If the list is not empty, collect the linked elements, starting with first-supplier
    if list_size > 0:
        try:
            ol_first_chan_id = ol_list_id.FindByAttribute("first-supplier", 0).ConvertToIdentifier()
            supplier_list = read_ol_supplier_list(ol_first_chan_id.ConvertToIdentifier())
            return supplier_list
        except AttributeError:
            print("ERROR: Couldn't read first-supplier")
            return []

#################


# Create the Soar agent
kernel = sml.Kernel.CreateKernelInNewThread(12121)
agent = kernel.CreateAgent("My Agent")

# Load agent productions
print(f"Loading agent files from: {AGENT_PATH}")
agent.LoadProductions(AGENT_PATH)


## Set up input for this agent

il_id = agent.GetInputLink()

il_priorities_id = il_id.CreateIdWME("priorities")
il_priorities_id.CreateFloatWME("total-cost", 11.01)
il_priorities_id.CreateIntWME("sustainability", 11)
il_priorities_id.CreateIntWME("quality", 11)
il_priorities_id.CreateIntWME("availability", 8)
il_priorities_id.CreateIntWME("packaging", 8)
il_priorities_id.CreateIntWME("speed", 7)

il_settings_id = il_id.CreateIdWME("settings")
il_settings_id.CreateIntWME("max-output-suppliers", 3)

ch01 = MyInputSupplier("supplier01")
ch01.set_totals(35.0, 12, 2)
ch01.set_ranks(3,3,1,3,2)
ch01.push_to_input_link(il_id)

ch02 = MyInputSupplier("supplier02")
ch02.set_totals(35.0, 9, 2)
ch02.set_ranks(2,2,3,1,1)
ch02.push_to_input_link(il_id)

ch03 = MyInputSupplier("supplier03")
ch03.set_totals(25.0, 13, 1)
ch03.set_ranks(3,3,3,3,1)
ch03.push_to_input_link(il_id)

ch04 = MyInputSupplier("supplier04")
ch04.set_totals(25.0, 11, 2)
ch04.set_ranks(3,3,2,2,1)
ch04.push_to_input_link(il_id)

ch05 = MyInputSupplier("supplier05")
ch05.set_totals(25.0, 14, 1)
ch05.set_ranks(3,3,3,3,2)
ch05.push_to_input_link(il_id)

ch06 = MyInputSupplier("supplier06")
ch06.set_totals(35.0, 10, 0)
ch06.set_ranks(2,3,1,3,1)
ch06.push_to_input_link(il_id)

# Run the agent on this input
print("Running Soar!")
agent.RunSelfForever()    # Runs the agent until it interrupts itself
agent.RunSelf(1)          # Runs the agent for one more decision cycle so that the output is transmitted

# Collect output from Soar
recommendation_list = get_supplier_list(agent)
print(f"Recommended suppliers: {recommendation_list}")


# STEP 7.1: Modify the input for this agent so that supplier04 is now the most expensive
ch04.set_totals(45.0, 11, 2)
ch04.push_to_input_link(il_id)

# STEP 7.2: Run the agent again on this modified input
print("Running Soar!")
agent.RunSelfForever()    # Runs the agent until it interrupts itself
agent.RunSelf(1)          # Runs the agent for one more decision cycle so that the output is transmitted

# STEP 7.3: Re-collect output from the agent
recommendation_list = get_supplier_list(agent)
print(f"Newly recommended suppliers: {recommendation_list}")


# Close the kernel before ending the script. Always a good practice!
print("\nGoodnight Soar.")
kernel.Shutdown()
