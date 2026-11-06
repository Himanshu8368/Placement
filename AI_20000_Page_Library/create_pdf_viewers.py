#!/usr/bin/env python3
"""
Create PDF viewer pages for all PDF files in the docs directory.
This script finds all PDF files and creates corresponding .md files with embedded PDF viewers.
"""

import os
from pathlib import Path

def create_pdf_viewer_page(pdf_path, relative_pdf_path):
    """Create a markdown file with embedded PDF viewer for a PDF file."""
    
    # Get PDF filename without extension
    pdf_name = pdf_path.stem
    viewer_md_path = pdf_path.parent / f"{pdf_name}.pdf.md"
    
    # Create relative path for PDF (relative to docs/)
    pdf_url = str(relative_pdf_path).replace("\\", "/")
    
    # Create the viewer page content
    viewer_content = f"""---
title: {pdf_name} (PDF Viewer)
---

# {pdf_name}

<div class="pdf-viewer-container" style="width: 100%; height: 80vh; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; margin: 20px 0;">
    <iframe 
        src="{pdf_url}" 
        style="width: 100%; height: 100%; border: none;"
        type="application/pdf"
        title="{pdf_name} PDF Viewer">
        <p>Your browser does not support PDFs. 
        <a href="{pdf_url}" download>Download the PDF</a> instead.</p>
    </iframe>
</div>

<div style="margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 4px;">
    <h3>📥 Download Options</h3>
    <a href="{pdf_url}" download class="md-button md-button--primary" style="margin-right: 10px;">
        Download PDF
    </a>
    <a href="{pdf_url}" target="_blank" class="md-button">
        Open in New Tab
    </a>
</div>

<style>
    @media (max-width: 768px) {{
        .pdf-viewer-container {{
            height: 60vh;
        }}
    }}
</style>

<script>
// Fallback for browsers that don't support iframe PDF viewing
if (navigator.userAgent.indexOf('Safari') > -1 && navigator.userAgent.indexOf('Chrome') === -1) {{
    // Safari may need special handling
    document.querySelector('iframe').src = "{pdf_url}";
}}
</script>

---

*PDF file: `{pdf_name}.pdf`*
"""
    
    # Write the viewer page
    with open(viewer_md_path, 'w', encoding='utf-8') as f:
        f.write(viewer_content)
    
    return viewer_md_path

def find_and_create_viewers(docs_path="docs"):
    """Find all PDF files and create viewer pages."""
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        print(f"❌ Error: {docs_path} directory not found")
        return []
    
    pdf_files = []
    viewer_pages = []
    
    # Find all PDF files
    for pdf_file in docs_dir.rglob("*.pdf"):
        # Skip if viewer page already exists
        viewer_md = pdf_file.parent / f"{pdf_file.stem}.pdf.md"
        if viewer_md.exists():
            print(f"⏭️  Skipping {pdf_file.name} (viewer already exists)")
            continue
        
        # Get relative path from docs/
        relative_path = pdf_file.relative_to(docs_dir)
        
        # Create viewer page
        viewer_path = create_pdf_viewer_page(pdf_file, relative_path)
        pdf_files.append(pdf_file)
        viewer_pages.append(viewer_path)
        
        print(f"✅ Created viewer: {viewer_path.relative_to(docs_dir)}")
    
    return pdf_files, viewer_pages

def main():
    print("🔍 Searching for PDF files in docs/ directory...\n")
    
    pdf_files, viewer_pages = find_and_create_viewers()
    
    print(f"\n📊 Summary:")
    print(f"   Found {len(pdf_files)} PDF file(s)")
    print(f"   Created {len(viewer_pages)} viewer page(s)")
    
    if viewer_pages:
        print(f"\n💡 Next steps:")
        print(f"   1. Run: python generate_nav.py")
        print(f"   2. Update mkdocs.yml navigation")
        print(f"   3. Run: mkdocs build")
        print(f"\n✅ PDF viewer pages are ready!")

if __name__ == "__main__":
    main()

