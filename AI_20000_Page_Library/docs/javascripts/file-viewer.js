// Multi-Format File Viewer System
(function() {
    'use strict';
    
    // File type configurations
    const fileViewers = {
        '.pdf': {
            type: 'pdf',
            viewer: 'iframe',
            mimeType: 'application/pdf'
        },
        '.txt': {
            type: 'text',
            viewer: 'text',
            mimeType: 'text/plain'
        },
        '.ppt': {
            type: 'powerpoint',
            viewer: 'office',
            mimeType: 'application/vnd.ms-powerpoint'
        },
        '.pptx': {
            type: 'powerpoint',
            viewer: 'office',
            mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        },
        '.doc': {
            type: 'word',
            viewer: 'office',
            mimeType: 'application/msword'
        },
        '.docx': {
            type: 'word',
            viewer: 'office',
            mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        },
        '.xls': {
            type: 'excel',
            viewer: 'office',
            mimeType: 'application/vnd.ms-excel'
        },
        '.xlsx': {
            type: 'excel',
            viewer: 'office',
            mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        },
        '.md': {
            type: 'markdown',
            viewer: 'markdown',
            mimeType: 'text/markdown'
        }
    };
    
    function getFileExtension(filename) {
        const parts = filename.split('.');
        return parts.length > 1 ? '.' + parts[parts.length - 1].toLowerCase() : '';
    }
    
    function getFileViewerConfig(filePath) {
        const ext = getFileExtension(filePath);
        return fileViewers[ext] || null;
    }
    
    function createTextFileViewer(fileUrl, fileName) {
        return fetch(fileUrl)
            .then(response => response.text())
            .then(text => {
                const viewer = document.createElement('div');
                viewer.className = 'file-viewer-container text-viewer';
                viewer.innerHTML = `
                    <div class="file-viewer-toolbar">
                        <a href="${fileUrl}" download class="md-button md-button--primary">
                            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                                <path d="M14 2v6h6"/>
                                <path d="M16 13H8"/>
                                <path d="M16 17H8"/>
                                <path d="M10 9H8"/>
                            </svg>
                            Download ${fileName}
                        </a>
                        <button onclick="copyTextContent()" class="md-button">
                            <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
                            </svg>
                            Copy Text
                        </button>
                    </div>
                    <div class="file-viewer-content text-content">
                        <pre><code>${escapeHtml(text)}</code></pre>
                    </div>
                `;
                return viewer;
            })
            .catch(error => {
                return createErrorViewer('Failed to load text file: ' + error.message);
            });
    }
    
    function createOfficeFileViewer(fileUrl, fileName, fileType) {
        const viewer = document.createElement('div');
        viewer.className = 'file-viewer-container office-viewer';
        
        // Use Microsoft Office Online Viewer or Google Docs Viewer
        const officeViewerUrl = `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(window.location.origin + fileUrl)}`;
        const googleViewerUrl = `https://docs.google.com/viewer?url=${encodeURIComponent(window.location.href.replace(/\/[^\/]*$/, fileUrl))}&embedded=true`;
        
        viewer.innerHTML = `
            <div class="file-viewer-toolbar">
                <a href="${fileUrl}" download class="md-button md-button--primary">
                    <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M16 13H8"/>
                        <path d="M16 17H8"/>
                        <path d="M10 9H8"/>
                    </svg>
                    Download ${fileName}
                </a>
                <a href="${fileUrl}" target="_blank" class="md-button">
                    <svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                        <polyline points="15 3 21 3 21 9"/>
                        <line x1="10" y1="14" x2="21" y2="3"/>
                    </svg>
                    Open in New Tab
                </a>
                <div class="file-viewer-note" style="margin-left: auto; padding: 0.5rem 1rem; background: var(--md-default-fg-color--lightest); border-radius: 4px; font-size: 0.875rem;">
                    💡 Tip: If the file doesn't load, download it and open with Microsoft Office or Google Docs
                </div>
            </div>
            <div class="file-viewer-wrapper">
                <iframe 
                    src="${officeViewerUrl}" 
                    class="file-iframe"
                    title="Viewer for ${fileName}"
                    style="width: 100%; height: 80vh; min-height: 600px; border: none;">
                    <p>Your browser does not support iframes. 
                    <a href="${fileUrl}" download>Download the file</a> instead.</p>
                </iframe>
            </div>
        `;
        
        return Promise.resolve(viewer);
    }
    
    function createErrorViewer(message) {
        const viewer = document.createElement('div');
        viewer.className = 'file-viewer-container error-viewer';
        viewer.innerHTML = `
            <div class="file-viewer-error">
                <svg style="width: 48px; height: 48px; color: var(--md-default-fg-color--lighter); margin-bottom: 1rem;" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                </svg>
                <p>${escapeHtml(message)}</p>
                <p style="font-size: 0.875rem; color: var(--md-default-fg-color--lighter);">
                    Please try downloading the file instead.
                </p>
            </div>
        `;
        return Promise.resolve(viewer);
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Global function for copying text
    window.copyTextContent = function() {
        const textContent = document.querySelector('.text-content pre code')?.textContent;
        if (textContent) {
            navigator.clipboard.writeText(textContent).then(() => {
                const btn = event.target.closest('button');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<svg style="width: 16px; height: 16px; vertical-align: middle; margin-right: 5px;" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>Copied!';
                setTimeout(() => {
                    btn.innerHTML = originalText;
                }, 2000);
            });
        }
    };
    
    // Auto-detect and create viewers for file links
    function initializeFileViewers() {
        // Check if we're on a file viewer page
        const article = document.querySelector('.md-content__inner') || document.querySelector('article');
        if (!article) return;
        
        // Look for file links in the page
        const fileLinks = article.querySelectorAll('a[href$=".pdf"], a[href$=".txt"], a[href$=".ppt"], a[href$=".pptx"], a[href$=".doc"], a[href$=".docx"], a[href$=".xls"], a[href$=".xlsx"]');
        
        fileLinks.forEach(link => {
            const href = link.getAttribute('href');
            const config = getFileViewerConfig(href);
            
            if (config && config.viewer === 'text') {
                // Add click handler for text files
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    // Could open in modal or new page
                });
            }
        });
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeFileViewers);
    } else {
        initializeFileViewers();
    }
})();
