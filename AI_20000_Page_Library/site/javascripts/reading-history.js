// Reading History Tracking
(function() {
    'use strict';
    
    const STORAGE_KEY = 'ai_compendium_history';
    const MAX_HISTORY = 10;
    
    function getHistory() {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    }
    
    function saveHistory(history) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    }
    
    function addToHistory(title, url) {
        const history = getHistory();
        
        // Remove if already exists
        const filtered = history.filter(h => h.url !== url);
        
        // Add to beginning
        filtered.unshift({
            title: title,
            url: url,
            date: new Date().toISOString(),
            timestamp: Date.now()
        });
        
        // Keep only last MAX_HISTORY items
        const trimmed = filtered.slice(0, MAX_HISTORY);
        saveHistory(trimmed);
    }
    
    function trackPage() {
        const title = document.querySelector('h1')?.textContent || document.title;
        const url = window.location.href;
        
        // Track page view
        addToHistory(title, url);
        
        // Track reading progress
        let lastScrollPosition = 0;
        let readingStartTime = Date.now();
        
        function updateReadingProgress() {
            const scrollPercent = (window.pageYOffset / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
            
            // Save progress every 5 seconds or when scrolling significantly
            if (Math.abs(scrollPercent - lastScrollPosition) > 10 || Date.now() - readingStartTime > 5000) {
                const history = getHistory();
                const current = history.find(h => h.url === url);
                if (current) {
                    current.progress = Math.min(scrollPercent, 100);
                    current.lastRead = new Date().toISOString();
                    saveHistory(history);
                }
                lastScrollPosition = scrollPercent;
                readingStartTime = Date.now();
            }
        }
        
        window.addEventListener('scroll', updateReadingProgress, { passive: true });
        
        // Save on page unload
        window.addEventListener('beforeunload', updateReadingProgress);
    }
    
    function createHistoryWidget() {
        // Only show on homepage
        if (!window.location.pathname.endsWith('/') && !window.location.pathname.endsWith('/index.html')) {
            return;
        }
        
        const history = getHistory();
        if (history.length === 0) return;
        
        const widget = document.createElement('div');
        widget.className = 'reading-history-widget';
        widget.style.cssText = `
            margin: 2rem 0;
            padding: 1.5rem;
            background: var(--md-default-fg-color--lightest);
            border-radius: 8px;
            border: 1px solid var(--md-default-fg-color--lighter);
        `;
        
        widget.innerHTML = `
            <h2 style="margin-top: 0;">📖 Recently Read</h2>
            <ul style="list-style: none; padding: 0; margin: 0;">
                ${history.slice(0, 3).map(item => {
                    const date = new Date(item.date);
                    const progress = item.progress ? ` (${Math.round(item.progress)}% read)` : '';
                    return `
                        <li style="margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--md-default-fg-color--lighter);">
                            <a href="${item.url}" style="text-decoration: none; color: var(--md-primary-fg-color);">
                                ${item.title}
                            </a>
                            <span style="display: block; font-size: 0.875rem; color: var(--md-default-fg-color--lighter); margin-top: 0.25rem;">
                                ${date.toLocaleDateString()}${progress}
                            </span>
                        </li>
                    `;
                }).join('')}
            </ul>
            ${history.length > 3 ? `<p style="margin-top: 1rem; margin-bottom: 0; font-size: 0.875rem; color: var(--md-default-fg-color--lighter);">${history.length - 3} more pages in history</p>` : ''}
        `;
        
        const mainContent = document.querySelector('.md-content__inner') || document.querySelector('main');
        if (mainContent) {
            const firstHeading = mainContent.querySelector('h1');
            if (firstHeading) {
                firstHeading.insertAdjacentElement('afterend', widget);
            }
        }
    }
    
    // Track current page
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            trackPage();
            createHistoryWidget();
        });
    } else {
        trackPage();
        createHistoryWidget();
    }
})();
