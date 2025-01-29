# docs/assets/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    initializeNavigation();
    initializeCodeBlocks();
    initializeThemeToggle();
    new DocSearch();
});

function initializeNavigation() {
    // Expand/collapse navigation sections
    const navSections = document.querySelectorAll('.nav-section > h3');
    navSections.forEach(section => {
        section.addEventListener('click', () => {
            const content = section.nextElementSibling;
            section.classList.toggle('expanded');
            content.style.maxHeight = content.style.maxHeight ? null : `${content.scrollHeight}px`;
        });
    });

    // Highlight current page in navigation
    const currentPath = window.location.pathname;
    const currentNavItem = document.querySelector(`.site-nav a[href$="${currentPath}"]`);
    if (currentNavItem) {
        currentNavItem.closest('li').classList.add('active');
        const parentSection = currentNavItem.closest('.nav-section');
        if (parentSection) {
            parentSection.querySelector('h3').classList.add('expanded');
        }
    }
}

function initializeCodeBlocks() {
    // Add copy button to code blocks
    document.querySelectorAll('pre code').forEach(block => {
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.textContent = 'Copy';
        
        copyButton.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(block.textContent);
                copyButton.textContent = 'Copied!';
                setTimeout(() => {
                    copyButton.textContent = 'Copy';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code:', err);
            }
        });

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        wrapper.appendChild(copyButton);
    });
}

function initializeThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-theme');
            const isDark = document.body.classList.contains('dark-theme');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });

        // Set initial theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
        }
    }
}