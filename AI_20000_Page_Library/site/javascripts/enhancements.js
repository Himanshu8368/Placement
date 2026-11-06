// Reading Progress Bar
(function() {
    'use strict';
    
    // Create progress bar
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    document.body.appendChild(progressBar);
    
    // Update progress on scroll
    function updateProgress() {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollPercent = (scrollTop / (documentHeight - windowHeight)) * 100;
        progressBar.style.width = scrollPercent + '%';
    }
    
    window.addEventListener('scroll', updateProgress);
    window.addEventListener('resize', updateProgress);
    updateProgress();
})();

// Reading Time Calculator
(function() {
    'use strict';
    
    function calculateReadingTime() {
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return;
        
        const text = article.innerText || article.textContent;
        const wordCount = text.trim().split(/\s+/).length;
        const readingTime = Math.ceil(wordCount / 200); // Average reading speed: 200 words per minute
        
        // Check if reading time already exists
        if (document.querySelector('.reading-time')) return;
        
        const readingTimeEl = document.createElement('div');
        readingTimeEl.className = 'reading-time';
        readingTimeEl.textContent = `~${readingTime} min read`;
        
        const title = article.querySelector('h1');
        if (title) {
            title.insertAdjacentElement('afterend', readingTimeEl);
        }
    }
    
    // Calculate when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', calculateReadingTime);
    } else {
        calculateReadingTime();
    }
})();

// Keyboard Shortcuts
(function() {
    'use strict';
    
    const shortcuts = {
        '/': 'Focus search',
        's': 'Focus search',
        '?': 'Show keyboard shortcuts',
        'Esc': 'Close dialogs',
        'g h': 'Go to homepage',
        'g n': 'Next page',
        'g p': 'Previous page'
    };
    
    let shortcutsVisible = false;
    
    function createShortcutsModal() {
        const modal = document.createElement('div');
        modal.className = 'keyboard-shortcuts';
        modal.innerHTML = `
            <h3>⌨️ Keyboard Shortcuts</h3>
            <ul style="list-style: none; padding: 0; margin: 0;">
                ${Object.entries(shortcuts).map(([key, desc]) => `
                    <li style="margin-bottom: 0.5rem; display: flex; justify-content: space-between;">
                        <kbd>${key}</kbd>
                        <span>${desc}</span>
                    </li>
                `).join('')}
            </ul>
            <button onclick="this.parentElement.classList.remove('show')" 
                    style="margin-top: 1rem; padding: 0.5rem 1rem; background: var(--md-primary-fg-color); 
                           color: white; border: none; border-radius: 4px; cursor: pointer; width: 100%;">
                Close
            </button>
        `;
        document.body.appendChild(modal);
        return modal;
    }
    
    let shortcutsModal = null;
    
    document.addEventListener('keydown', function(e) {
        // Show shortcuts with ?
        if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            if (!shortcutsModal) {
                shortcutsModal = createShortcutsModal();
            }
            shortcutsModal.classList.toggle('show');
            shortcutsVisible = !shortcutsVisible;
        }
        
        // Focus search with / or s
        if ((e.key === '/' || e.key === 's') && !e.ctrlKey && !e.metaKey && !e.altKey) {
            const target = e.target;
            if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                const searchInput = document.querySelector('input[type="search"]') || 
                                  document.querySelector('.md-search__input');
                if (searchInput) {
                    searchInput.focus();
                }
            }
        }
        
        // Close modals with Esc
        if (e.key === 'Escape') {
            if (shortcutsModal) {
                shortcutsModal.classList.remove('show');
            }
        }
    });
})();

// Smooth Scroll to Top Enhancement
(function() {
    'use strict';
    
    // Enhance the existing back-to-top button
    const backToTopBtn = document.querySelector('.md-top');
    if (backToTopBtn) {
        backToTopBtn.addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
})();

// Print Page Functionality
(function() {
    'use strict';
    
    function addPrintButton() {
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return;
        
        const printBtn = document.createElement('button');
        printBtn.className = 'md-button md-button--primary';
        printBtn.style.cssText = 'margin: 1rem 0; display: inline-flex; align-items: center; gap: 0.5rem;';
        printBtn.innerHTML = `
            <svg style="width: 16px; height: 16px;" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 8H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3zm-3 11H8v-5h8v5zm3-7c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm-1-9H6v4h12V3z"/>
            </svg>
            Print Page
        `;
        printBtn.onclick = () => window.print();
        
        const title = article.querySelector('h1');
        if (title) {
            title.insertAdjacentElement('afterend', printBtn);
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addPrintButton);
    } else {
        addPrintButton();
    }
})();

// Copy Code Block Enhancement
(function() {
    'use strict';
    
    // Add copy button to code blocks if not already present
    document.querySelectorAll('pre code').forEach(function(codeBlock) {
        const pre = codeBlock.parentElement;
        if (pre.querySelector('.copy-button')) return; // Already has copy button
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-button';
        copyBtn.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: var(--md-default-bg-color);
            border: 1px solid var(--md-default-fg-color--lighter);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        copyBtn.textContent = 'Copy';
        copyBtn.onmouseover = () => copyBtn.style.opacity = '1';
        copyBtn.onmouseout = () => copyBtn.style.opacity = '0.7';
        
        copyBtn.onclick = function() {
            const text = codeBlock.textContent;
            navigator.clipboard.writeText(text).then(function() {
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            });
        };
        
        pre.style.position = 'relative';
        pre.appendChild(copyBtn);
    });
})();
