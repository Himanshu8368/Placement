---
title: PDF Viewer Template
---

# PDF Viewer Template

This template shows how PDF files are displayed in the website.

## How PDFs Work

When you place a PDF file in any Part or Section folder:

1. **Automatic Detection**: The `generate_nav.py` script automatically finds PDF files
2. **Viewer Page Creation**: A `.pdf.md` file is created next to each PDF
3. **Navigation Integration**: PDFs appear in the navigation with "(PDF)" suffix
4. **Embedded Viewing**: PDFs are displayed directly in the website using iframe

## Adding PDFs

### Method 1: Automatic (Recommended)

1. Place your PDF file in any folder:
   ```
   docs/Part_01_Mathematics_for_AI/Section_01_Linear_Algebra/Notes.pdf
   ```

2. Run the navigation generator:
   ```bash
   python generate_nav.py
   ```

3. This automatically:
   - Creates `Notes.pdf.md` viewer page
   - Adds it to navigation structure
   - Updates nav_structure.txt

4. Update mkdocs.yml with new navigation

### Method 2: Manual

1. Place PDF file in desired location
2. Create a `.pdf.md` file next to it with this template:

```markdown
---
title: Your PDF Name (PDF Viewer)
---

# Your PDF Name

<div class="pdf-viewer-container">
    <div class="pdf-viewer-toolbar">
        <a href="YourPDF.pdf" download class="md-button md-button--primary">
            Download PDF
        </a>
        <a href="YourPDF.pdf" target="_blank" class="md-button">
            Open in New Tab
        </a>
    </div>
    
    <div class="pdf-viewer-wrapper">
        <iframe 
            src="YourPDF.pdf" 
            class="pdf-iframe"
            type="application/pdf"
            title="PDF Viewer">
            <p>Your browser does not support PDFs. 
            <a href="YourPDF.pdf" download>Download the PDF</a> instead.</p>
        </iframe>
    </div>
</div>

<style>
.pdf-viewer-container {
    margin: 20px 0;
    border: 1px solid var(--md-default-fg-color--lighter);
    border-radius: 4px;
    overflow: hidden;
    background: var(--md-default-bg-color);
}

.pdf-viewer-toolbar {
    padding: 12px 16px;
    background: var(--md-default-fg-color--lightest);
    border-bottom: 1px solid var(--md-default-fg-color--lighter);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}

.pdf-viewer-wrapper {
    width: 100%;
    height: 80vh;
    min-height: 600px;
    position: relative;
    background: #525252;
}

.pdf-iframe {
    width: 100%;
    height: 100%;
    border: none;
    display: block;
}

.md-button {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    text-decoration: none;
    border-radius: 0.2rem;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

.md-button--primary {
    background-color: #4051b5;
    color: white;
}

.md-button--primary:hover {
    background-color: #303fa1;
}

.md-button:not(.md-button--primary) {
    background-color: var(--md-default-bg-color);
    color: var(--md-default-fg-color);
    border: 1px solid var(--md-default-fg-color--lighter);
}

.md-button:not(.md-button--primary):hover {
    background-color: var(--md-default-fg-color--lightest);
}

@media (max-width: 768px) {
    .pdf-viewer-wrapper {
        height: 70vh;
        min-height: 500px;
    }
    
    .pdf-viewer-toolbar {
        flex-direction: column;
        align-items: stretch;
    }
    
    .md-button {
        width: 100%;
        justify-content: center;
        margin-bottom: 8px;
    }
}
</style>
```

## Features

- ✅ **Embedded PDF Viewer** - View PDFs directly in the website
- ✅ **Download Button** - Easy download option
- ✅ **Open in New Tab** - Full-screen viewing option
- ✅ **Responsive Design** - Works on mobile devices
- ✅ **Dark Mode Compatible** - Adapts to theme
- ✅ **Automatic Navigation** - PDFs appear in site navigation

## Browser Support

- **Chrome/Edge**: Full support with native PDF viewer
- **Firefox**: Full support with native PDF viewer
- **Safari**: Full support with native PDF viewer
- **Mobile**: Works on iOS and Android browsers

## Example Structure

```
docs/
└── Part_01_Mathematics_for_AI/
    └── Section_01_Linear_Algebra/
        ├── Vector_Spaces.md          # Regular markdown
        ├── Lecture_Notes.pdf         # PDF file
        └── Lecture_Notes.pdf.md       # Auto-generated viewer
```

In navigation, it appears as:
- Section 01: Linear Algebra
  - Vector Spaces
  - Lecture Notes (PDF) ← Click to view PDF

## Troubleshooting

**PDF doesn't display:**
- Check that PDF file path is correct
- Ensure PDF file exists in the same directory
- Try opening PDF in new tab to verify it's not corrupted

**Navigation doesn't show PDF:**
- Run `python generate_nav.py` again
- Check that `.pdf.md` file was created
- Verify navigation in `nav_structure.txt`

**PDF loads slowly:**
- Large PDFs may take time to load
- Consider splitting large PDFs into smaller files
- Use "Open in New Tab" for better performance
