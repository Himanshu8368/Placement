// Bookmarking System
(function() {
    'use strict';
    
    const STORAGE_KEY = 'ai_compendium_bookmarks';
    
    function getBookmarks() {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    }
    
    function saveBookmarks(bookmarks) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
    }
    
    function isBookmarked(url) {
        const bookmarks = getBookmarks();
        return bookmarks.some(b => b.url === url);
    }
    
    function addBookmark(title, url) {
        const bookmarks = getBookmarks();
        if (!isBookmarked(url)) {
            bookmarks.push({
                title: title,
                url: url,
                date: new Date().toISOString()
            });
            saveBookmarks(bookmarks);
            return true;
        }
        return false;
    }
    
    function removeBookmark(url) {
        const bookmarks = getBookmarks();
        const filtered = bookmarks.filter(b => b.url !== url);
        saveBookmarks(filtered);
    }
    
    function toggleBookmark(title, url) {
        if (isBookmarked(url)) {
            removeBookmark(url);
            return false;
        } else {
            addBookmark(title, url);
            return true;
        }
    }
    
    function createBookmarkButton() {
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return;
        
        const title = document.querySelector('h1')?.textContent || document.title;
        const url = window.location.href;
        
        const bookmarkBtn = document.createElement('button');
        bookmarkBtn.className = 'bookmark-btn md-button';
        bookmarkBtn.style.cssText = `
            margin: 1rem 0;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            position: relative;
        `;
        
        function updateButton() {
            const bookmarked = isBookmarked(url);
            bookmarkBtn.innerHTML = `
                <svg style="width: 16px; height: 16px;" fill="currentColor" viewBox="0 0 24 24">
                    <path d="${bookmarked ? 
                        'M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z' : 
                        'M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2zm0 15l-5-2.18L7 18V5h10v13z'
                    }"/>
                </svg>
                ${bookmarked ? 'Bookmarked' : 'Bookmark'}
            `;
            bookmarkBtn.style.backgroundColor = bookmarked ? '#4051b5' : 'transparent';
            bookmarkBtn.style.color = bookmarked ? 'white' : 'var(--md-default-fg-color)';
        }
        
        bookmarkBtn.onclick = function() {
            const added = toggleBookmark(title, url);
            updateButton();
            showNotification(added ? 'Page bookmarked!' : 'Bookmark removed');
        };
        
        updateButton();
        
        const titleEl = article.querySelector('h1');
        if (titleEl) {
            titleEl.insertAdjacentElement('afterend', bookmarkBtn);
        }
    }
    
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--md-default-bg-color);
            border: 1px solid var(--md-default-fg-color--lighter);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }
    
    // Add CSS animations
    if (!document.getElementById('bookmark-styles')) {
        const style = document.createElement('style');
        style.id = 'bookmark-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createBookmarkButton);
    } else {
        createBookmarkButton();
    }
})();
