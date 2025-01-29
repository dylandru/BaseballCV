# docs/assets/js/search.js
class DocSearch {
    constructor() {
        this.searchIndex = null;
        this.searchInput = document.getElementById('search');
        this.searchResults = document.getElementById('search-results');
        this.initializeSearch();
    }

    async initializeSearch() {
        try {
            const response = await fetch('/BaseballCV/assets/js/search-index.json');
            this.searchIndex = await response.json();
            this.bindSearchEvents();
        } catch (error) {
            console.error('Error initializing search:', error);
        }
    }

    bindSearchEvents() {
        this.searchInput.addEventListener('input', this.debounce(() => {
            this.performSearch(this.searchInput.value);
        }, 300));

        // Close search results when clicking outside
        document.addEventListener('click', (event) => {
            if (!event.target.closest('.search-container')) {
                this.hideSearchResults();
            }
        });
    }

    performSearch(query) {
        if (!query) {
            this.hideSearchResults();
            return;
        }

        const results = this.searchIndex.filter(page => {
            const searchableText = `${page.title} ${page.content}`.toLowerCase();
            return searchableText.includes(query.toLowerCase());
        }).slice(0, 5);

        this.displayResults(results);
    }

    displayResults(results) {
        if (!results.length) {
            this.searchResults.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }

        const resultsHTML = results.map(result => `
            <a href="${result.url}" class="search-result">
                <h4>${result.title}</h4>
                <p>${this.highlightText(result.excerpt, this.searchInput.value)}</p>
            </a>
        `).join('');

        this.searchResults.innerHTML = resultsHTML;
        this.showSearchResults();
    }

    highlightText(text, query) {
        if (!query) return text;
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    showSearchResults() {
        this.searchResults.style.display = 'block';
    }

    hideSearchResults() {
        this.searchResults.style.display = 'none';
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}