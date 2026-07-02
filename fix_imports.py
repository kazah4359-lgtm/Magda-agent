import os

files_to_fix = [
    "magda_agent/integration/mcp_exporter_v9.py",
    "magda_agent/integration/mcp_exporter.py"
]

for filename in files_to_fix:
    with open(filename, "r") as f:
        content = f.read()

    content = content.replace("from magda_agent.skills.mcp_export import MagdaMCPAdapter", "from magda_agent.integration.mcp_export import MCPExporter")
    # Replace MagdaMCPAdapter with MCPExporter in the rest of the file
    content = content.replace("MagdaMCPAdapter", "MCPExporter")

    with open(filename, "w") as f:
        f.write(content)
