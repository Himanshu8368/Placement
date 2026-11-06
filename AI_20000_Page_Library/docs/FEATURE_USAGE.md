# Feature Usage Guide

This guide explains how to use the new features added to the AI & Machine Learning Compendium website.

## 🔗 Related Pages Suggestions

### What it does:
Automatically shows related pages at the bottom of each article, helping you discover related content.

### How it works:
- **Automatic Detection**: The system automatically finds related pages based on:
  - Pages in the same section
  - Pages in the same part
  - Navigation structure

### Features:
- Shows up to 5 related pages
- Appears at the bottom of articles
- Click any link to navigate to related content
- Visual indicator with icon

### Customization:
To manually specify related pages for a specific article, add meta tags in the markdown frontmatter:

```yaml
---
title: Your Page Title
related-pages:
  - title: Related Page 1
    url: /path/to/page1/
  - title: Related Page 2
    url: /path/to/page2/
---
```

## 📖 Glossary Popup System

### What it does:
Provides instant definitions for technical terms by hovering over them in articles.

### How it works:
- **Automatic Detection**: Glossary terms are automatically highlighted in articles
- **Hover to See Definition**: Hover over any highlighted term to see its definition
- **Visual Indicators**: Terms are highlighted with a dotted underline

### Supported Terms:
The glossary includes 50+ key AI/ML terms including:
- Activation Function
- Backpropagation
- CNN (Convolutional Neural Network)
- Deep Learning
- Transformer
- And many more...

### Features:
- **Instant Tooltips**: Hover over terms to see definitions
- **Smart Highlighting**: Terms are automatically detected and highlighted
- **Non-intrusive**: Doesn't affect code blocks or special formatting
- **Responsive**: Tooltips position themselves intelligently

### How to Use:
1. Read any article
2. Look for terms with a dotted underline
3. Hover your mouse over the term
4. A tooltip will appear with the definition
5. Move your mouse away to hide the tooltip

### Adding New Terms:
To add new glossary terms, edit `docs/javascripts/glossary.js` and add entries to the `glossary` object:

```javascript
const glossary = {
    'your term': 'Definition of your term',
    // ... existing terms
};
```

## 💾 Bookmarking System

### What it does:
Save your favorite pages for quick access later.

### How to Use:
1. Visit any page
2. Click the "Bookmark" button (appears below the page title)
3. The button changes to "Bookmarked" ✓
4. Click again to remove the bookmark

### Features:
- Bookmarks are saved in your browser
- Works offline (localStorage)
- Visual feedback when bookmarking
- Persistent across sessions

## 📚 Reading History

### What it does:
Tracks pages you've visited and shows your reading progress.

### How to Use:
- **Automatic Tracking**: History is tracked automatically as you browse
- **View on Homepage**: Visit the homepage to see "Recently Read" section
- **Progress Tracking**: Shows reading progress percentage for each page

### Features:
- Shows last 3 recently read pages on homepage
- Tracks reading progress (scroll percentage)
- Displays last read date
- Keeps history of up to 50 pages

## ⌨️ Keyboard Shortcuts

Press `?` to see all available keyboard shortcuts:

- `/` or `s` - Focus search bar
- `?` - Show/hide keyboard shortcuts
- `Esc` - Close dialogs
- `g h` - Go to homepage
- `g n` - Next page
- `g p` - Previous page

## 📊 Reading Progress Bar

### What it does:
Shows your reading progress at the top of the page.

### Features:
- Visual progress bar at the top
- Updates as you scroll
- Helps track position in long articles
- Smooth animation

## ⏱️ Reading Time Estimates

### What it does:
Shows estimated reading time for each article.

### Features:
- Automatically calculated
- Based on 200 words per minute
- Appears below page title
- Helps plan reading sessions

## 🖨️ Print-Friendly Pages

### What it does:
Optimizes pages for printing and PDF export.

### How to Use:
1. Click the "Print Page" button on any article
2. Use browser's print dialog (Ctrl+P / Cmd+P)
3. Pages are optimized for printing

### Features:
- Removes navigation and sidebars
- Focuses on content
- Clean, readable output
- Optimized page breaks

---

*For more information about all features, see [FEATURES.md](FEATURES.md)*
