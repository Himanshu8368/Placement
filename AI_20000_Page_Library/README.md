# AI & Machine Learning Compendium

A comprehensive 20,000-page compilation covering all aspects of AI and Machine Learning.

> 📖 **For complete setup instructions and detailed documentation, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

## Structure

This compendium is organized into 11 major parts:

1. **Part 01**: Mathematics for AI
2. **Part 02**: Classical Machine Learning
3. **Part 03**: Deep Learning Fundamentals
4. **Part 04**: Computer Vision
5. **Part 05**: Natural Language Processing
6. **Part 06**: Large Language Models
7. **Part 07**: Reinforcement Learning
8. **Part 08**: Robotics and Ethics
9. **Part 09**: MLOps and Systems
10. **Part 10**: Specialized AI
11. **Part 11**: AI Strategy and Future

## Website (MkDocs)

### Setup

1. Install MkDocs and Material theme:
pip install mkdocs mkdocs-material
2. Serve locally:
mkdocs serve
3. Build static site:
mkdocs build4. Deploy to GitHub Pages:
mkdocs gh-deploy### Features

**Core:**
- Full-text search across all 20,000 pages
- Beautiful Material Design theme
- MathJax support for mathematical equations
- Mobile-responsive design
- Dark/Light mode toggle
- PDF viewer with embedded viewing

**Reading & Navigation:**
- Reading progress bar
- Reading time estimates
- Reading history tracking
- Bookmarking system
- Related pages suggestions
- Glossary popup system (50+ terms)
- Keyboard shortcuts

**Content:**
- Print-friendly pages
- Enhanced typography
- Code block copy buttons
- Interactive tooltips

## LaTeX/PDF Generation

### Prerequisites

1. Install Pandoc:
   - Windows: Download from [pandoc.org](https://pandoc.org/installing.html)
   - macOS: `brew install pandoc`
   - Linux: `sudo apt-get install pandoc`

2. Install LaTeX distribution:
   - Windows: [MiKTeX](https://miktex.org/) or [TeX Live](https://www.tug.org/texlive/)
   - macOS: [MacTeX](https://www.tug.org/mactex/)
   - Linux: `sudo apt-get install texlive-full`

### Usage

Generate complete PDF:ash
python generate_latex.py --output Complete_AI_CompendiumGenerate PDF for specific part:
python generate_latex.py --part Part_01_Mathematics_for_AI --output Part01_MathematicsGenerate LaTeX only (no PDF compilation):
python generate_latex.py --latex-only --output My_Book## EPUB Generation

Use the existing script:
python manage_compendium.py --all
python manage_compendium.py --part Part_01_Mathematics_for_AI## Search

Search across all content:sh
python manage_compendium.py --search "neural network"## License

[Your License Here]