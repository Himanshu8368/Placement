// Related Pages Suggestions System
(function() {
    'use strict';
    
    // Get related pages from page metadata or auto-detect
    function getRelatedPages() {
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return [];
        
        // Try to get from meta tags or data attributes
        const metaTags = document.querySelectorAll('meta[name="related-pages"]');
        if (metaTags.length > 0) {
            return Array.from(metaTags).map(tag => ({
                title: tag.getAttribute('content'),
                url: tag.getAttribute('data-url')
            })).filter(p => p.title && p.url);
        }
        
        // Auto-detect based on current page path
        return autoDetectRelatedPages();
    }
    
    function autoDetectRelatedPages() {
        const currentPath = window.location.pathname;
        const currentUrl = window.location.href;
        
        // Extract part and section from URL
        const pathParts = currentPath.split('/').filter(p => p);
        const related = [];
        
        // Get all navigation links
        const navLinks = document.querySelectorAll('.md-nav__link[href]');
        const allPages = Array.from(navLinks).map(link => ({
            title: link.textContent.trim(),
            url: link.getAttribute('href'),
            element: link
        })).filter(p => p.url && p.url !== currentPath);
        
        // Find pages in same section
        if (pathParts.length >= 2) {
            const sectionPath = pathParts.slice(0, 2).join('/');
            const sameSection = allPages.filter(p => {
                const pParts = p.url.split('/').filter(x => x);
                return pParts.length >= 2 && pParts.slice(0, 2).join('/') === sectionPath;
            }).slice(0, 3);
            related.push(...sameSection);
        }
        
        // Find pages in same part
        if (pathParts.length >= 1) {
            const partPath = pathParts[0];
            const samePart = allPages.filter(p => {
                const pParts = p.url.split('/').filter(x => x);
                return pParts.length >= 1 && pParts[0] === partPath && !related.some(r => r.url === p.url);
            }).slice(0, 2);
            related.push(...samePart);
        }
        
        // Limit to 5 related pages
        return related.slice(0, 5);
    }
    
    function createRelatedPagesWidget() {
        const related = getRelatedPages();
        if (related.length === 0) return;
        
        const widget = document.createElement('div');
        widget.className = 'related-pages-widget';
        widget.style.cssText = `
            margin: 3rem 0 2rem 0;
            padding: 1.5rem;
            background: var(--md-default-fg-color--lightest);
            border-radius: 8px;
            border-left: 4px solid #4051b5;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        `;
        
        widget.innerHTML = `
            <h3 style="margin-top: 0; margin-bottom: 1rem; color: var(--md-primary-fg-color); display: flex; align-items: center; gap: 0.5rem;">
                <svg style="width: 20px; height: 20px;" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                Related Pages
            </h3>
            <ul style="list-style: none; padding: 0; margin: 0;">
                ${related.map(page => `
                    <li style="margin-bottom: 0.75rem;">
                        <a href="${page.url}" 
                           style="text-decoration: none; color: var(--md-primary-fg-color); 
                                  display: flex; align-items: center; gap: 0.5rem;
                                  padding: 0.5rem; border-radius: 4px;
                                  transition: background-color 0.2s;">
                            <svg style="width: 16px; height: 16px; flex-shrink: 0;" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/>
                            </svg>
                            <span>${page.title}</span>
                        </a>
                    </li>
                `).join('')}
            </ul>
        `;
        
        // Add hover effect
        widget.querySelectorAll('a').forEach(link => {
            link.addEventListener('mouseenter', function() {
                this.style.backgroundColor = 'var(--md-default-bg-color)';
            });
            link.addEventListener('mouseleave', function() {
                this.style.backgroundColor = 'transparent';
            });
        });
        
        // Insert before footer or at end of article
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (article) {
            // Try to insert before footer or hr
            const footer = article.querySelector('footer') || article.querySelector('hr');
            if (footer) {
                footer.insertAdjacentElement('beforebegin', widget);
            } else {
                article.appendChild(widget);
            }
        }
    }
    
    // Create widget when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createRelatedPagesWidget);
    } else {
        createRelatedPagesWidget();
    }
})();
