import os
from pathlib import Path

def create_pdf_viewer_page(pdf_file, md_file_path, pdf_relative_path):
    """Create a markdown page that embeds a PDF viewer using PDF.js"""
    pdf_name = pdf_file.stem.replace("_", " ").replace(".", " ")
    # Use relative path for PDF.js viewer
    pdf_url = f"/{pdf_relative_path}"
    
    pdf_content = f"""# {pdf_name}

<div class="pdf-viewer-container">
    <div class="pdf-viewer-toolbar">
        <a href="{pdf_url}" download class="md-button md-button--primary" style="margin-right: 10px;">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download PDF
        </a>
        <a href="{pdf_url}" target="_blank" class="md-button" style="margin-right: 10px;">
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

<style>
.pdf-viewer-container {{
    margin: 20px 0;
    border: 1px solid var(--md-default-fg-color--lighter);
    border-radius: 4px;
    overflow: hidden;
    background: var(--md-default-bg-color);
}}

.pdf-viewer-toolbar {{
    padding: 12px 16px;
    background: var(--md-default-fg-color--lightest);
    border-bottom: 1px solid var(--md-default-fg-color--lighter);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
}}

.pdf-viewer-wrapper {{
    width: 100%;
    height: 80vh;
    min-height: 600px;
    position: relative;
    background: #525252;
}}

.pdf-iframe {{
    width: 100%;
    height: 100%;
    border: none;
    display: block;
}}

.md-button {{
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 0.2rem;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}}

.md-button--primary {{
    background-color: #4051b5;
    color: white;
}}

.md-button--primary:hover {{
    background-color: #303fa1;
}}

.md-button:not(.md-button--primary) {{
    background-color: var(--md-default-bg-color);
    color: var(--md-default-fg-color);
    border: 1px solid var(--md-default-fg-color--lighter);
}}

.md-button:not(.md-button--primary):hover {{
    background-color: var(--md-default-fg-color--lightest);
}}

@media (max-width: 768px) {{
    .pdf-viewer-wrapper {{
        height: 70vh;
        min-height: 500px;
    }}
    
    .pdf-viewer-toolbar {{
        flex-direction: column;
        align-items: stretch;
    }}
    
    .md-button {{
        width: 100%;
        justify-content: center;
        margin-bottom: 8px;
    }}
}}
</style>

---
*PDF Document - Use the toolbar above to download or open in a new tab for the best viewing experience*
"""
    # Only create if it doesn't exist
    if not md_file_path.exists():
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(pdf_content)

def generate_navigation():
    """Generate hierarchical navigation structure for mkdocs.yml"""
    docs_path = Path("docs")
    nav_lines = ["nav:", "  - Home: README.md"]
    
    # Get all parts
    parts = sorted([d for d in docs_path.iterdir() 
                   if d.is_dir() and d.name.startswith("Part_")])
    
    for part in parts:
        part_name = part.name.replace("_", " ").replace("Part ", "Part ")
        nav_lines.append(f'  - "{part_name}":')
        
        # Check for PDFs in part root
        part_pdfs = sorted([f for f in part.iterdir() 
                           if f.is_file() and f.suffix == ".pdf"])
        
        for pdf_file in part_pdfs:
            rel_path = pdf_file.relative_to(docs_path)
            rel_path_str = str(rel_path).replace("\\", "/")
            pdf_name = pdf_file.stem.replace("_", " ").replace(".", " ")
            pdf_md_path = pdf_file.with_suffix('.pdf.md')
            create_pdf_viewer_page(pdf_file, pdf_md_path, rel_path_str)
            nav_lines.append(f'    - "{pdf_name} (PDF)": {str(pdf_md_path.relative_to(docs_path)).replace("\\", "/")}')
        
        # Get all sections in this part
        sections = sorted([d for d in part.iterdir() 
                          if d.is_dir() and d.name.startswith("Section_")])
        
        for section in sections:
            # Extract section number and name properly
            section_parts = section.name.split("_")
            if len(section_parts) >= 3:
                section_num = section_parts[1]
                section_name = " ".join(section_parts[2:]).replace("_", " ")
                section_title = f"Section {section_num}: {section_name}"
            else:
                section_title = section.name.replace("Section_", "").replace("_", " ")
            
            nav_lines.append(f'    - "{section_title}":')
            
            # Get all markdown files in this section
            md_files = sorted([f for f in section.iterdir() 
                              if f.is_file() and f.suffix == ".md" and f.name != "README.md" and not f.name.endswith(".pdf.md")])
            
            for md_file in md_files:
                rel_path = md_file.relative_to(docs_path)
                # Convert Windows path to forward slashes for YAML
                rel_path_str = str(rel_path).replace("\\", "/")
                chapter_name = md_file.stem.replace("_", " ")
                nav_lines.append(f'      - {chapter_name}: {rel_path_str}')
            
            # Get all PDF files in this section
            pdf_files = sorted([f for f in section.iterdir() 
                               if f.is_file() and f.suffix == ".pdf"])
            
            for pdf_file in pdf_files:
                rel_path = pdf_file.relative_to(docs_path)
                rel_path_str = str(rel_path).replace("\\", "/")
                pdf_name = pdf_file.stem.replace("_", " ").replace(".", " ")
                # Create a markdown wrapper for PDF viewing
                pdf_md_path = pdf_file.with_suffix('.pdf.md')
                create_pdf_viewer_page(pdf_file, pdf_md_path, rel_path_str)
                nav_lines.append(f'      - "{pdf_name} (PDF)": {str(pdf_md_path.relative_to(docs_path)).replace("\\", "/")}')
    
    return "\n".join(nav_lines)

if __name__ == "__main__":
    nav = generate_navigation()
    print(nav)
    # Write to a file for review
    with open("nav_structure.txt", "w", encoding="utf-8") as f:
        f.write(nav)
    print("\n✅ Navigation generated! PDF viewer pages created automatically.")
