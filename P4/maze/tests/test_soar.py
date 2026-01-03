#!/usr/bin/env python3
"""Test Soar loading step by step"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Add Soar to path
sys.path.append(r"C:\Users\marce\Desktop\euu\cuarto\robotica\SoarSuite_9.6.4-Multiplatform\bin")

print("Step 1: Importing Soar...")
import Python_sml_ClientInterface as sml
print("[OK] Import successful")

print("\nStep 2: Creating kernel...")
kernel = sml.Kernel.CreateKernelInNewThread()
print(f"[OK] Kernel created: {kernel}")

print("\nStep 3: Creating agent...")
agent = kernel.CreateAgent("TestAgent")
print(f"[OK] Agent created: {agent}")

print("\nStep 4: Loading productions...")
agent_path = Path(__file__).parent.parent / "agent" / "maze-navigator-dfs.soar"
print(f"Loading from: {agent_path}")
result = agent.LoadProductions(str(agent_path))

if result:
    print("[OK] Productions loaded successfully")
else:
    print(f"[ERROR] Error loading productions: {agent.GetLastErrorDescription()}")
    sys.exit(1)

print("\nStep 5: Getting input link...")
input_link = agent.GetInputLink()
print(f"[OK] Input link: {input_link}")

print("\nStep 6: Running agent for 1 decision cycle...")
kernel.RunAllAgents(1)
print("[OK] Agent ran successfully")

print("\n[SUCCESS] All tests passed!")
kernel.Shutdown()
