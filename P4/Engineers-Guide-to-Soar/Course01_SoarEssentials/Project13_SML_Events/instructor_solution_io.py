"""This file separates our helper classes and output parsing functions for App4"""

import random

import Python_sml_ClientInterface as sml

#########################################################################
# Input Helper Classes:

class MyInputSupplier:
    """A class for defining an input candidate-supplier and adding it to an agent input-link"""
    def __init__(self, name) -> None:
        self.name = name
        self.total_cost = 0.0
        self.total_score = 0
        self.total_sats = 0
        self.sustainability = 0
        self.availability = 0
        self.quality = 0
        self.packaging = 0
        self.speed = 0

        self.il_cand_chan_id = None

    def set_totals(self, total_cost, total_score, total_sats):
        self.total_cost = total_cost
        self.total_score = total_score
        self.total_sats = total_sats

    def set_ranks(self, sustainability, availability, quality, packaging, speed):
        self.sustainability = sustainability
        self.availability = availability
        self.quality = quality
        self.packaging = packaging
        self.speed = speed

    def push_to_input_link(self, input_link_id):
        # If this supplier has already been created on the input-link, destroy it before making a new one
        self.remove_from_input_link()

        # Add this data as a new input supplier
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

    def remove_from_input_link(self):
        """Destroy all input WMEs for this supplier"""
        # When the root WME is destroyed, all orphaned child WMEs are also destroyed.
        if self.il_cand_chan_id != None:
            self.il_cand_chan_id.DestroyWME()
            self.il_cand_chan_id = None


class MyAgentInputManager:
    """A class for managing all desired WME structures for our Soar agent.
    An instance of this class can be easily referenced within callback functions.
    """
    def __init__(self, agent) -> None:
        self.agent = agent
        self.input_supplier_list = []

        il_id = agent.GetInputLink()

        self.il_priorities_id = il_id.CreateIdWME("priorities")
        self.il_priorities_id.CreateFloatWME("total-cost", 11.01)
        self.il_priorities_id.CreateIntWME("sustainability", 11)
        self.il_priorities_id.CreateIntWME("quality", 11)
        self.il_priorities_id.CreateIntWME("availability", 8)
        self.il_priorities_id.CreateIntWME("packaging", 8)
        self.il_priorities_id.CreateIntWME("speed", 7)

        self.il_settings_id = il_id.CreateIdWME("settings")
        self.il_settings_id.CreateIntWME("max-output-suppliers", 3)

    def add_random_supplier(self):
        chan = MyInputSupplier(f"supplier-{len(self.input_supplier_list)+1}")
        self.input_supplier_list.append(chan)

        chan.set_totals(
            total_cost=random.uniform(0.0, 40.0), 
            total_score=random.randint(1,5),
            total_sats=random.randint(1,2)
        )
        
        chan.set_ranks(
            sustainability=random.randint(1,5),
            availability=random.randint(1,5),
            quality=random.randint(1,5),
            packaging=random.randint(1,5),
            speed=random.randint(1,5)
        )

        chan.push_to_input_link(self.agent.GetInputLink())

    def clear_suppliers(self):
        for chan in self.input_supplier_list:
            chan.remove_from_input_link()
        self.input_supplier_list.clear()

    def clear_input(self):
        if self.il_priorities_id != None:
            self.il_priorities_id.DestroyWME()
            self.il_priorities_id = None

        if self.il_settings_id != None:
            self.il_settings_id.DestroyWME()
            self.il_settings_id = None

        self.clear_suppliers()


#########################################################################
# Output Parsing Functions:

def read_ol_supplier_list(ol_supplier_id):
    return_list = []
    try:
        # Recursively follow "^next" links in the returned list to read the recommended list of suppliers
        for i in range(ol_supplier_id.GetNumberChildren()):
            # Loop through all supplier augmentations and process each in a case-based manner
            ol_list_item_wme = ol_supplier_id.GetChild(i)
            attr = ol_list_item_wme.GetAttribute()

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


def parse_supplier_list(ol_list_id):
    # Check the size of the output list
    try:
        list_size = ol_list_id.FindByAttribute("count", 0).ConvertToIntElement().GetValue()
    except AttributeError:
        # print("ERROR: No count attribute in output supplier-list!")
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
