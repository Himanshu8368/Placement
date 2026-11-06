# File Viewer Guide

This guide explains how to view different file types on the AI & Machine Learning Compendium website.

## Supported File Types

The website supports viewing the following file types:

### 📄 PDF Files (.pdf)

- **Viewer**: Native browser PDF viewer
- **Features**:
  - Embedded viewing
  - Download option
  - Open in new tab
- **Best for**: Documents, presentations, research papers

### 📝 Text Files (.txt)

- **Viewer**: Text content display
- **Features**:
  - Readable text display
  - Copy to clipboard button
  - Download option
- **Best for**: Code snippets, notes, plain text documents

### 📊 PowerPoint Files (.ppt, .pptx)

- **Viewer**: Microsoft Office Online Viewer
- **Features**:
  - Embedded presentation viewer
  - Download option
  - Open in new tab
- **Best for**: Presentations, slideshows
- **Note**: Requires internet connection for online viewer

### 📄 Word Documents (.doc, .docx)

- **Viewer**: Microsoft Office Online Viewer
- **Features**:
  - Embedded document viewer
  - Download option
  - Open in new tab
- **Best for**: Documents, reports, articles

### 📈 Excel Files (.xls, .xlsx)

- **Viewer**: Microsoft Office Online Viewer
- **Features**:
  - Embedded spreadsheet viewer
  - Download option
  - Open in new tab
- **Best for**: Data tables, spreadsheets, calculations

## How to Use

### Viewing Files

1. **Automatic Detection**: Files are automatically detected and viewer pages are created
2. **Navigation**: File viewers appear in navigation with file type labels (e.g., "Document (PDF)")
3. **Viewer Page**: Click on a file in navigation to open its viewer page
4. **Toolbar Actions**: Use toolbar buttons to download or open files

### Adding Files

1. **Place File**: Put your file in any Part or Section folder
2. **Run Generator**: Run `python generate_file_viewers.py` to create viewer pages
3. **Update Navigation**: Run `python generate_nav.py` to update navigation
4. **Rebuild Site**: Run `mkdocs build` to rebuild the site

## Viewer Features

### Common Features (All File Types)

- ✅ Download button
- ✅ Open in new tab option
- ✅ Responsive design
- ✅ Mobile-friendly

### Text File Specific

- ✅ Copy to clipboard button
- ✅ Syntax highlighting (for code)
- ✅ Readable formatting

### Office File Specific

- ✅ Online viewer (Microsoft Office Online)
- ✅ Fallback to download if viewer fails
- ✅ Tips for best viewing experience

## Troubleshooting

### File Doesn't Load

- **Office Files**: Download and open with Microsoft Office or Google Docs
- **PDF Files**: Try opening in a new tab
- **Text Files**: Check file encoding (should be UTF-8)

### Viewer Not Appearing

- Ensure file is in `docs/` directory
- Run `python generate_file_viewers.py`
- Check file extension is supported
- Rebuild site with `mkdocs build`

### Office Viewer Issues

- Office Online Viewer requires internet connection
- Some files may need to be downloaded
- Try Google Docs Viewer as alternative

## File Size Recommendations

- **PDF**: Up to 50MB recommended
- **Text**: Up to 5MB recommended
- **Office Files**: Up to 25MB recommended

## Best Practices

1. **File Naming**: Use descriptive names without special characters
2. **File Organization**: Group related files in same folder
3. **File Size**: Keep files reasonably sized for web viewing
4. **File Format**: Use standard formats (.pdf, .txt, .pptx, etc.)

---

*For technical details, see the file viewer implementation in `docs/javascripts/file-viewer.js`*
