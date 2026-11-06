import os
import subprocess
import argparse
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class CompendiumManager:
    def __init__(self, folder_name="docs"):
        # Use docs folder as the source
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.library_path = os.path.join(script_dir, folder_name)
        
        if not os.path.exists(self.library_path):
            print(f"[ERROR] Could not find library at {self.library_path}")
            sys.exit(1)
            
        self.parts = sorted([d for d in os.listdir(self.library_path) if os.path.isdir(os.path.join(self.library_path, d)) and d.startswith("Part_")])

    def search(self, query):
        """Search for a keyword across all 20,000 pages."""
        print(f"[SEARCH] Searching for: '{query}' across the library...\n")
        results_found = 0
        
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                results_found += 1
                                # Get relative path for cleaner display
                                rel_path = os.path.relpath(file_path, self.library_path)
                                print(f"[MATCH] Found in [{rel_path}] (Line {i+1}):")
                                print(f"   > ...{line.strip()}...\n")
        
        if results_found == 0:
            print("No matches found.")
        else:
            print(f"[SUCCESS] Search Complete. Found {results_found} occurrences.")

    def _get_metadata(self, title):
        return f"---\ntitle: {title}\nauthor: AI Master Compendium\nlanguage: en-US\n---\n\n"

    def compile(self, target_parts=None, target_section=None, output_name="Master_Book"):
        temp_md = "temp_compile.md"
        output_epub = f"{output_name}.epub"
        
        with open(temp_md, "w", encoding="utf-8") as master:
            master.write(self._get_metadata(output_name.replace("_", " ")))
            parts_to_process = [p for p in self.parts if p == target_parts] if target_parts else self.parts
            
            for part in parts_to_process:
                master.write(f"\n# {part.replace('_', ' ')}\n")
                part_path = os.path.join(self.library_path, part)
                sections = sorted([d for d in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, d))])
                
                if target_section:
                    sections = [s for s in sections if target_section in s]

                for sec in sections:
                    master.write(f"\n## {sec.replace('_', ' ')}\n")
                    sec_path = os.path.join(part_path, sec)
                    chapters = sorted([f for f in os.listdir(sec_path) if f.endswith(".md")])
                    for chapter in chapters:
                        with open(os.path.join(sec_path, chapter), "r", encoding="utf-8") as f:
                            content = f.read().replace("# ", "### ")
                            master.write(content + "\n\n")

        print(f"[INFO] Compiling {output_epub}...")
        try:
            subprocess.run([
                "pandoc", temp_md, "-o", output_epub,
                "--toc", "--toc-depth=3",
                "--epub-chapter-level=2",
                "--mathjax"
            ], check=True)
            print(f"[SUCCESS] File saved as {output_epub}")
        except FileNotFoundError:
            print("[ERROR] Pandoc not found. Please install Pandoc to convert to ePub.")
        finally:
            if os.path.exists(temp_md):
                os.remove(temp_md)

def main():
    parser = argparse.ArgumentParser(description="AI Library Master Controller")
    parser.add_argument("--all", action="store_true", help="Compile the whole directory")
    parser.add_argument("--part", type=str, help="Compile a specific part")
    parser.add_argument("--section", type=str, help="Compile a specific section")
    parser.add_argument("--search", type=str, help="Search for a topic across all files")
    
    args = parser.parse_args()
    manager = CompendiumManager()

    if args.search:
        manager.search(args.search)
    elif args.all:
        manager.compile(output_name="Complete_AI_Library")
    elif args.part and args.section:
        manager.compile(target_parts=args.part, target_section=args.section, output_name=f"{args.section}_DeepDive")
    elif args.part:
        manager.compile(target_parts=args.part, output_name=f"{args.part}_Volume")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()