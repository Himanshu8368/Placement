import os
import subprocess
import argparse
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class LaTeXGenerator:
    def __init__(self, library_path="docs"):
        self.library_path = Path(library_path).resolve()
        if not self.library_path.exists():
            print(f"[ERROR] Library path not found: {self.library_path}")
            sys.exit(1)
        
        self.parts = sorted([d for d in self.library_path.iterdir() 
                           if d.is_dir() and d.name.startswith("Part_")])
        
    def get_all_markdown_files(self, part_path):
        """Recursively get all markdown files in a part."""
        md_files = []
        for root, dirs, files in os.walk(part_path):
            for file in files:
                if file.endswith(".md") and file != "README.md":
                    md_files.append(Path(root) / file)
        return sorted(md_files)
    
    def convert_md_to_latex_content(self, md_file):
        """Convert markdown file to LaTeX content."""
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Basic markdown to LaTeX conversions
        # Headers
        content = content.replace("# ", "\\section{")
        content = content.replace("## ", "\\subsection{")
        content = content.replace("### ", "\\subsubsection{")
        
        # Close headers (simple approach - may need refinement)
        lines = content.split('\n')
        result = []
        for line in lines:
            if line.startswith("\\section{") or line.startswith("\\subsection{") or line.startswith("\\subsubsection{"):
                if not line.endswith("}"):
                    line = line.rstrip() + "}\n"
            result.append(line)
        
        return '\n'.join(result)
    
    def generate_latex(self, target_parts=None, output_name="Master_Book", 
                      template_path="latex_template.tex"):
        """Generate LaTeX document from markdown files."""
        
        # Read template
        if not os.path.exists(template_path):
            print(f"[ERROR] Template not found: {template_path}")
            sys.exit(1)
        
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        # Determine which parts to process
        parts_to_process = [p for p in self.parts 
                          if target_parts is None or p.name == target_parts]
        
        # Generate content
        latex_content = []
        
        for part in parts_to_process:
            part_title = part.name.replace("_", " ")
            latex_content.append(f"\\parttitle{{{part_title}}}\n")
            
            # Get all markdown files in this part
            md_files = self.get_all_markdown_files(part)
            
            for md_file in md_files:
                rel_path = md_file.relative_to(self.library_path)
                print(f"  Processing: {rel_path}")
                
                # Use pandoc for better conversion
                try:
                    result = subprocess.run(
                        ["pandoc", str(md_file), "-f", "markdown", "-t", "latex"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    latex_content.append(result.stdout)
                    latex_content.append("\n\\newpage\n")
                except subprocess.CalledProcessError as e:
                    print(f"    [WARNING] Pandoc conversion failed for {md_file}")
                    print(f"    Error: {e.stderr}")
                    # Fallback: basic conversion
                    content = self.convert_md_to_latex_content(md_file)
                    latex_content.append(content)
                    latex_content.append("\n\\newpage\n")
        
        # Insert content into template
        template = template.replace("% Content will be inserted here by the script", 
                                   "\n".join(latex_content))
        
        # Write LaTeX file
        output_latex = f"{output_name}.tex"
        with open(output_latex, "w", encoding="utf-8") as f:
            f.write(template)
        
        print(f"\n[SUCCESS] LaTeX file generated: {output_latex}")
        return output_latex
    
    def compile_pdf(self, latex_file, output_name=None):
        """Compile LaTeX to PDF."""
        if output_name is None:
            output_name = latex_file.replace(".tex", "")
        
        print(f"\n[INFO] Compiling PDF: {latex_file}...")
        print("   This may take several minutes for large documents...\n")
        
        # Run pdflatex (twice for proper references)
        for run in [1, 2]:
            print(f"   Running pdflatex (pass {run}/2)...")
            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", 
                     f"-jobname={output_name}", latex_file],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"[ERROR] Error during compilation (pass {run}):")
                    print(result.stdout[-1000:])  # Last 1000 chars
                    return False
            except FileNotFoundError:
                print("[ERROR] pdflatex not found. Please install a LaTeX distribution:")
                print("   - Windows: MiKTeX or TeX Live")
                print("   - macOS: MacTeX")
                print("   - Linux: texlive-full")
                return False
        
        # Clean up auxiliary files
        aux_files = [".aux", ".log", ".out", ".toc", ".lof", ".lot"]
        for ext in aux_files:
            aux_file = f"{output_name}{ext}"
            if os.path.exists(aux_file):
                os.remove(aux_file)
        
        pdf_file = f"{output_name}.pdf"
        if os.path.exists(pdf_file):
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"\n[SUCCESS] PDF generated successfully: {pdf_file}")
            print(f"   File size: {size_mb:.2f} MB")
            return True
        else:
            print("[ERROR] PDF file not found after compilation")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX/PDF from AI Compendium"
    )
    parser.add_argument(
        "--part", 
        type=str, 
        help="Generate for specific part (e.g., Part_01_Mathematics_for_AI)"
    )
    parser.add_argument(
        "--latex-only", 
        action="store_true", 
        help="Generate LaTeX file only (don't compile PDF)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="AI_Compendium",
        help="Output filename (without extension)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="latex_template.tex",
        help="Path to LaTeX template file"
    )
    
    args = parser.parse_args()
    
    generator = LaTeXGenerator()
    
    # Generate LaTeX
    latex_file = generator.generate_latex(
        target_parts=args.part,
        output_name=args.output,
        template_path=args.template
    )
    
    # Compile PDF if requested
    if not args.latex_only:
        generator.compile_pdf(latex_file, args.output)
    else:
        print("\n[INFO] To compile PDF, run:")
        print(f"   pdflatex {latex_file}")

if __name__ == "__main__":
    main()