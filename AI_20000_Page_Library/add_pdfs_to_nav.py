#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to automatically find PDF files and add them to navigation.
Run this after adding PDF files to your docs folder.
"""
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def find_all_pdfs(docs_path="docs"):
    """Find all PDF files in the docs directory."""
    pdf_files = []
    docs_dir = Path(docs_path)
    
    for pdf_file in docs_dir.rglob("*.pdf"):
        rel_path = pdf_file.relative_to(docs_dir)
        pdf_files.append({
            'path': pdf_file,
            'rel_path': str(rel_path).replace("\\", "/"),
            'name': pdf_file.stem.replace("_", " "),
            'parent': pdf_file.parent
        })
    
    return sorted(pdf_files, key=lambda x: x['rel_path'])

def create_pdf_viewer_page(pdf_info):
    """Create a markdown page that embeds a PDF viewer."""
    pdf_file = pdf_info['path']
    pdf_name = pdf_info['name']
    pdf_url = pdf_info['rel_path']
    
    # Create .pdf.md file next to the PDF
    md_file = pdf_file.with_suffix('.pdf.md')
    
    pdf_content = f"""# {pdf_name}

<div class="pdf-viewer-container">
    <div class="pdf-viewer-toolbar">
        <a href="{pdf_url}" download class="md-button md-button--primary">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download PDF
        </a>
        <a href="{pdf_url}" target="_blank" class="md-button">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in New Tab
        </a>
    </div>
    
    <div class="pdf-viewer-wrapper">
        <iframe 
            src="{pdf_url}" 
            class="pdf-iframe"
            type="application/pdf"
            title="PDF Viewer for {pdf_name}">
            <p>Your browser does not support PDFs. 
            <a href="{pdf_url}" download>Download the PDF</a> instead.</p>
        </iframe>
    </div>
</div>

---
*PDF Document - Use the toolbar above to download or open in a new tab for the best viewing experience*
"""
    
    # Only create if it doesn't exist
    if not md_file.exists():
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(pdf_content)
        return True
    return False

def main():
    """Main function to process all PDFs."""
    print("[SCAN] Scanning for PDF files...")
    pdf_files = find_all_pdfs()
    
    if not pdf_files:
        print("[INFO] No PDF files found in docs/ directory")
        print("\nTo add PDFs:")
        print("   1. Place PDF files in any Part or Section folder")
        print("   2. Run this script again: python add_pdfs_to_nav.py")
        print("   3. Run: python generate_nav.py")
        print("   4. Update mkdocs.yml with new navigation")
        return
    
    print(f"[SUCCESS] Found {len(pdf_files)} PDF file(s):\n")
    
    created = 0
    existing = 0
    
    for pdf_info in pdf_files:
        print(f"  [PDF] {pdf_info['rel_path']}")
        if create_pdf_viewer_page(pdf_info):
            print(f"     [OK] Created viewer page: {pdf_info['rel_path']}.md")
            created += 1
        else:
            print(f"     [INFO] Viewer page already exists")
            existing += 1
    
    print(f"\n[SUMMARY]")
    print(f"   - New viewer pages created: {created}")
    print(f"   - Existing viewer pages: {existing}")
    print(f"   - Total PDFs: {len(pdf_files)}")
    
    print(f"\n[NEXT STEPS]")
    print(f"   1. Run: python generate_nav.py")
    print(f"   2. Review nav_structure.txt")
    print(f"   3. Update mkdocs.yml navigation section")
    print(f"   4. Run: mkdocs serve (to test)")

if __name__ == "__main__":
    main()

