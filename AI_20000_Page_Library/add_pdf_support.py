#!/usr/bin/env python3
"""
Script to add PDF files to MkDocs navigation and create PDF viewer pages
"""
import os
from pathlib import Path

def find_pdf_files(docs_path="docs"):
    """Find all PDF files in the docs directory."""
    pdf_files = []
    docs_dir = Path(docs_path)
    
    for pdf_file in docs_dir.rglob("*.pdf"):
        rel_path = pdf_file.relative_to(docs_dir)
        pdf_files.append({
            'path': str(rel_path).replace("\\", "/"),
            'name': pdf_file.stem.replace("_", " "),
            'full_path': pdf_file
        })
    
    return sorted(pdf_files, key=lambda x: x['path'])

def create_pdf_viewer_page(pdf_path, output_path):
    """Create an HTML page that embeds a PDF using PDF.js."""
    pdf_viewer_html = f"""---
title: {Path(pdf_path).stem.replace('_', ' ')}
---

# {Path(pdf_path).stem.replace('_', ' ')}

<div style="width: 100%; height: 80vh; border: 1px solid #ccc; border-radius: 4px;">
    <iframe 
        src="https://mozilla.github.io/pdf.js/web/viewer.html?file=/{pdf_path}" 
        style="width: 100%; height: 100%; border: none;"
        title="PDF Viewer">
    </iframe>
</div>

<div style="margin-top: 20px;">
    <a href="/{pdf_path}" download class="md-button md-button--primary">
        <span class="twemoji">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>
        </span>
        Download PDF
    </a>
</div>

<style>
.md-button {{
    display: inline-block;
    padding: 0.5rem 1rem;
    margin-top: 1rem;
    text-decoration: none;
    border-radius: 0.2rem;
}}
.md-button--primary {{
    background-color: #4051b5;
    color: white;
}}
.md-button--primary:hover {{
    background-color: #303fa1;
}}
</style>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pdf_viewer_html)
    
    print(f"Created PDF viewer page: {output_path}")

def main():
    """Main function to process PDF files."""
    docs_path = "docs"
    pdf_files = find_pdf_files(docs_path)
    
    if not pdf_files:
        print("No PDF files found in docs directory.")
        print("Place PDF files in any Part/Section folder to view them on the website.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf['path']}")
        
        # Create a viewer page next to the PDF
        pdf_dir = pdf['full_path'].parent
        viewer_path = pdf_dir / f"{pdf['full_path'].stem}_viewer.md"
        create_pdf_viewer_page(pdf['path'], viewer_path)
    
    print(f"\n✅ Created {len(pdf_files)} PDF viewer page(s)")
    print("\nNext steps:")
    print("1. Run: python generate_nav.py")
    print("2. Update mkdocs.yml navigation to include the PDF viewer pages")
    print("3. Rebuild: mkdocs build")

if __name__ == "__main__":
    main()

