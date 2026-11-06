# Adding Files to the Website

This guide explains how to add various file types (PDF, TXT, PPT, DOCX, etc.) to your website so they can be viewed directly in the browser.

## Quick Start

1. **Place your file** in any Part or Section folder in `docs/`
2. **Generate viewer pages**: Run `python generate_file_viewers.py`
3. **Update navigation**: Run `python generate_nav.py` (if needed)
4. **Rebuild site**: Run `mkdocs build`

## Supported File Types

| Extension | Type | Viewer | Notes |
|-----------|------|--------|-------|
| `.pdf` | PDF | Browser PDF viewer | Best support |
| `.txt` | Text | Text display | Copy button available |
| `.ppt` | PowerPoint | Office Online | Requires internet |
| `.pptx` | PowerPoint | Office Online | Requires internet |
| `.doc` | Word | Office Online | Requires internet |
| `.docx` | Word | Office Online | Requires internet |
| `.xls` | Excel | Office Online | Requires internet |
| `.xlsx` | Excel | Office Online | Requires internet |

## Step-by-Step Guide

### 1. Add Your File

Place your file in the appropriate folder:

```
docs/
└── Part_01_Mathematics_for_AI/
    └── Section_01_Linear_Algebra/
        ├── Lecture_Notes.pdf
        ├── Code_Examples.txt
        └── Presentation.pptx
```

### 2. Generate Viewer Pages

Run the file viewer generator:

```bash
python generate_file_viewers.py
```

This will:
- Find all supported files in `docs/`
- Create viewer pages (`.md` files) for each file
- Place viewer pages next to the original files

### 3. Update Navigation (Optional)

If you want files to appear in navigation, update `mkdocs.yml` manually or use:

```bash
python generate_nav.py
```

### 4. Rebuild Website

```bash
mkdocs build
mkdocs serve
```

## File Naming Best Practices

- ✅ Use descriptive names: `Introduction_to_ML.pdf`
- ✅ Avoid spaces: Use underscores or hyphens
- ✅ Keep names short: Under 50 characters
- ❌ Avoid special characters: `@`, `#`, `$`, etc.
- ❌ Avoid spaces in filenames

## Examples

### Example 1: Adding a PDF

1. Place `Research_Paper.pdf` in `docs/Part_01_Mathematics_for_AI/`
2. Run `python generate_file_viewers.py`
3. A viewer page `Research-Paper-viewer.md` is created
4. Access via: `/Part_01_Mathematics_for_AI/Research-Paper-viewer/`

### Example 2: Adding a Text File

1. Place `Code_Snippet.txt` in `docs/Part_02_Classical_Machine_Learning/`
2. Run `python generate_file_viewers.py`
3. A viewer page `Code-Snippet-viewer.md` is created
4. Features: Readable display + Copy button

### Example 3: Adding a PowerPoint

1. Place `Presentation.pptx` in any folder
2. Run `python generate_file_viewers.py`
3. A viewer page is created with Office Online viewer
4. Users can view slides directly in browser

## Viewer Features by Type

### PDF Viewer
- ✅ Embedded viewing
- ✅ Download button
- ✅ Open in new tab
- ✅ Full browser PDF controls

### Text Viewer
- ✅ Readable text display
- ✅ Copy to clipboard button
- ✅ Download option
- ✅ Syntax-friendly formatting

### Office File Viewer
- ✅ Microsoft Office Online viewer
- ✅ Download option
- ✅ Open in new tab
- ✅ Fallback message if viewer fails

## Troubleshooting

### File Not Appearing
- Check file is in `docs/` directory
- Verify file extension is supported
- Run `python generate_file_viewers.py` again
- Check file permissions

### Viewer Not Loading
- **Office Files**: Requires internet connection
- **PDF Files**: Check browser PDF support
- **Text Files**: Verify file encoding (UTF-8 recommended)

### Navigation Not Updated
- Manually add to `mkdocs.yml` nav section
- Or run `python generate_nav.py` to auto-generate

## File Size Recommendations

- **PDF**: Up to 50MB
- **Text**: Up to 5MB
- **Office Files**: Up to 25MB

Larger files may take longer to load.

## Advanced Usage

### Custom Viewer Pages

You can manually create viewer pages using this template:

```markdown
---
title: Your File Name (File Type Viewer)
---

# Your File Name

<div class="file-viewer-container">
    <div class="file-viewer-toolbar">
        <a href="../your-file.ext" download class="md-button md-button--primary">
            Download File
        </a>
    </div>
    <div class="file-viewer-wrapper">
        <!-- Viewer content here -->
    </div>
</div>
```

### Batch Processing

To process multiple files at once:

```bash
# Generate all viewers
python generate_file_viewers.py

# Then rebuild site
mkdocs build
```

---

*For more details, see [FILE_VIEWER_GUIDE.md](FILE_VIEWER_GUIDE.md)*
