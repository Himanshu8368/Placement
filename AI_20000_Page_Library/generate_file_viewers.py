#!/usr/bin/env python3
"""
Create file viewer pages for various file types (PDF, TXT, PPT, DOCX, etc.)
"""
import os
from pathlib import Path

# Supported file types and their viewers
SUPPORTED_TYPES = {
    '.pdf': {'viewer': 'pdf', 'label': 'PDF'},
    '.txt': {'viewer': 'text', 'label': 'Text'},
    '.ppt': {'viewer': 'office', 'label': 'PowerPoint'},
    '.pptx': {'viewer': 'office', 'label': 'PowerPoint'},
    '.doc': {'viewer': 'office', 'label': 'Word'},
    '.docx': {'viewer': 'office', 'label': 'Word'},
    '.xls': {'viewer': 'office', 'label': 'Excel'},
    '.xlsx': {'viewer': 'office', 'label': 'Excel'},
}

def create_file_viewer_page(file_path, md_file_path, file_relative_path, file_type):
    """Create a markdown page that embeds a file viewer"""
    file_name = file_path.stem.replace("_", " ").replace(".", " ")
    file_ext = file_path.suffix.lower()
    file_url = f"../{file_relative_path.replace('/', '/')}"
    
    # Get file type info
    type_info = SUPPORTED_TYPES.get(file_ext, {'viewer': 'download', 'label': 'File'})
    viewer_type = type_info['viewer']
    file_label = type_info['label']
    
    if viewer_type == 'pdf':
        # PDF viewer
        content = f"""---
title: {file_name} ({file_label} Viewer)
---

# {file_name}

<div class="file-viewer-container pdf-viewer">
    <div class="file-viewer-toolbar">
        <a href="{file_url}" download class="md-button md-button--primary">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download {file_label}
        </a>
        <a href="{file_url}" target="_blank" class="md-button">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in New Tab
        </a>
    </div>
    
    <div class="file-viewer-wrapper">
        <iframe 
            src="{file_url}" 
            class="file-iframe"
            type="application/pdf"
            title="{file_label} Viewer for {file_name}">
            <p>Your browser does not support PDFs. 
            <a href="{file_url}" download>Download the {file_label}</a> instead.</p>
        </iframe>
    </div>
</div>

---
*{file_label} Document - Use the toolbar above to download or open in a new tab for the best viewing experience*
"""
    
    elif viewer_type == 'text':
        # Text file viewer
        content = f"""---
title: {file_name} ({file_label} Viewer)
---

# {file_name}

<div class="file-viewer-container text-viewer">
    <div class="file-viewer-toolbar">
        <a href="{file_url}" download class="md-button md-button--primary">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download {file_label}
        </a>
        <button onclick="copyTextContent()" class="md-button">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
            Copy Text
        </button>
    </div>
    
    <div class="file-viewer-content text-content">
        <iframe 
            src="{file_url}" 
            class="text-iframe"
            title="{file_label} Viewer for {file_name}"
            style="width: 100%; height: 70vh; min-height: 400px; border: 1px solid var(--md-default-fg-color--lighter); border-radius: 4px;">
            <p>Your browser does not support iframes. 
            <a href="{file_url}" download>Download the {file_label} file</a> instead.</p>
        </iframe>
    </div>
</div>

<script>
function copyTextContent() {{
    fetch('{file_url}')
        .then(response => response.text())
        .then(text => {{
            navigator.clipboard.writeText(text).then(() => {{
                const btn = event.target.closest('button');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>Copied!';
                setTimeout(() => {{ btn.innerHTML = originalText; }}, 2000);
            }});
        }});
}}
</script>

---
*{file_label} Document - Use the toolbar above to download or copy the content*
"""
    
    elif viewer_type == 'office':
        # Office file viewer (PPT, DOCX, XLSX)
        office_viewer_url = f"https://view.officeapps.live.com/op/embed.aspx?src={file_url}"
        google_viewer_url = f"https://docs.google.com/viewer?url={file_url}&embedded=true"
        
        content = f"""---
title: {file_name} ({file_label} Viewer)
---

# {file_name}

<div class="file-viewer-container office-viewer">
    <div class="file-viewer-toolbar">
        <a href="{file_url}" download class="md-button md-button--primary">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download {file_label}
        </a>
        <a href="{file_url}" target="_blank" class="md-button">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in New Tab
        </a>
        <div class="file-viewer-note">
            💡 Tip: If the file doesn't load, download it and open with Microsoft Office or Google Docs
        </div>
    </div>
    
    <div class="file-viewer-wrapper">
        <iframe 
            src="{office_viewer_url}" 
            class="file-iframe"
            title="{file_label} Viewer for {file_name}"
            style="width: 100%; height: 80vh; min-height: 600px; border: none;">
            <p>Your browser does not support iframes. 
            <a href="{file_url}" download>Download the {file_label} file</a> instead.</p>
        </iframe>
    </div>
</div>

---
*{file_label} Document - Use the toolbar above to download or open in a new tab. For best results, download and open with Microsoft Office or Google Docs.*
"""
    
    else:
        # Generic download page
        content = f"""---
title: {file_name} ({file_label})
---

# {file_name}

<div class="file-viewer-container download-viewer">
    <div class="file-viewer-toolbar">
        <a href="{file_url}" download class="md-button md-button--primary">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                <path d="M14 2v6h6"/>
                <path d="M16 13H8"/>
                <path d="M16 17H8"/>
                <path d="M10 9H8"/>
            </svg>
            Download {file_label}
        </a>
        <a href="{file_url}" target="_blank" class="md-button">
            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            Open in New Tab
        </a>
    </div>
    
    <div class="file-viewer-info">
        <p>This file type is not directly viewable in the browser. Please download it to view.</p>
    </div>
</div>

---
*{file_label} File - Download to view*
"""
    
    # Write the viewer page
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def find_all_files(docs_path="docs", file_extensions=None):
    """Find all files with specified extensions"""
    if file_extensions is None:
        file_extensions = list(SUPPORTED_TYPES.keys())
    
    docs_dir = Path(docs_path)
    files = []
    
    for ext in file_extensions:
        for file_path in docs_dir.rglob(f"*{ext}"):
            if file_path.suffix.lower() == ext:
                rel_path = file_path.relative_to(docs_dir)
                files.append({
                    'path': file_path,
                    'rel_path': str(rel_path).replace("\\", "/"),
                    'ext': ext,
                    'name': file_path.stem
                })
    
    return sorted(files, key=lambda x: x['rel_path'])

def generate_file_viewers(docs_path="docs"):
    """Generate viewer pages for all supported file types"""
    files = find_all_files(docs_path)
    
    created = []
    for file_info in files:
        file_path = file_info['path']
        ext = file_info['ext']
        
        # Create viewer page name (avoid conflicts)
        viewer_name = file_path.stem.replace(" ", "-").replace(".", "-")
        md_file_path = file_path.parent / f"{viewer_name}-viewer.md"
        
        if create_file_viewer_page(file_path, md_file_path, file_info['rel_path'], ext):
            created.append({
                'file': file_info['rel_path'],
                'viewer': str(md_file_path.relative_to(docs_path)).replace("\\", "/"),
                'type': SUPPORTED_TYPES[ext]['label']
            })
    
    return created

if __name__ == "__main__":
    print("Generating file viewer pages...")
    created = generate_file_viewers()
    print(f"\n[SUCCESS] Created {len(created)} file viewer pages:")
    for item in created:
        print(f"  - {item['type']}: {item['file']} -> {item['viewer']}")
