#!/usr/bin/env python3
"""Update mkdocs.yml with new navigation structure"""

with open('nav_output_utf8.txt', 'r', encoding='utf-8') as f:
    new_nav = f.read()

with open('mkdocs.yml', 'r', encoding='utf-8') as f:
    mkdocs_content = f.read()

# Find the nav section
nav_start = mkdocs_content.find('nav:')
if nav_start == -1:
    print("Error: 'nav:' not found in mkdocs.yml")
    exit(1)

# Find where nav section ends (next top-level key or end of file)
nav_end = len(mkdocs_content)
for i in range(nav_start + 4, len(mkdocs_content)):
    if mkdocs_content[i] == '\n':
        # Check if next non-empty line is a top-level key (no indentation)
        j = i + 1
        while j < len(mkdocs_content) and mkdocs_content[j] in ' \t\n':
            j += 1
        if j < len(mkdocs_content):
            if mkdocs_content[j] not in ' \t-':
                nav_end = i + 1
                break

# Replace nav section
new_mkdocs = mkdocs_content[:nav_start] + new_nav + '\n'
if nav_end < len(mkdocs_content):
    new_mkdocs += mkdocs_content[nav_end:]

with open('mkdocs.yml', 'w', encoding='utf-8') as f:
    f.write(new_mkdocs)

print("✅ Successfully updated mkdocs.yml with hierarchical navigation structure")

