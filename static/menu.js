document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');

    // The menu is now positioned off-screen by default via CSS.
    // This script just handles the open/close classes.

    // Function to open the menu
    const openMenu = () => {
        sidebar.classList.add('open');
        mainContent.classList.add('sidebar-open');
    };

    // Function to close the menu
    const closeMenu = () => {
        sidebar.classList.remove('open');
        mainContent.classList.remove('sidebar-open');
    };

    // Event listener for the menu toggle button
    if (menuToggle) {
        menuToggle.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent this click from being caught by the document listener
            if (sidebar.classList.contains('open')) {
                closeMenu();
            } else {
                openMenu();
            }
        });
    }

    // Close menu when clicking outside of it
    document.addEventListener('click', (e) => {
        // If the sidebar is open and the click is not the toggle button or inside the sidebar
        if (sidebar.classList.contains('open') && !menuToggle.contains(e.target) && !sidebar.contains(e.target)) {
            closeMenu();
        }
    });

    // Display App Version from meta tag
    const versionMeta = document.querySelector('meta[name="app-version"]');
    if (versionMeta && versionMeta.content) {
        const appVersion = versionMeta.content;
        const versionElement = document.createElement('div');
        versionElement.className = 'app-version'; // For styling
        versionElement.textContent = `AudioMuse-AI - Version ${appVersion}`;
        sidebar.appendChild(versionElement);
    }

    /* --- Dark Mode Logic --- */
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const body = document.body;

    // Update toggle button text and ARIA state
    const updateToggleUI = (isDark) => {
        if (darkModeToggle) {
            darkModeToggle.innerHTML = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
            darkModeToggle.setAttribute('aria-pressed', isDark);
        }
    };

    // Check saved preference or system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Apply initial theme (also sync body class with html class set by FOUC prevention script)
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        body.classList.add('dark-mode');
        updateToggleUI(true);
    } else {
        // Remove dark-mode class if it was set by FOUC script but user actually prefers light
        body.classList.remove('dark-mode');
        document.documentElement.classList.remove('dark-mode');
        updateToggleUI(false);
    }

    // Toggle click handler
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', (e) => {
            e.preventDefault();
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            updateToggleUI(isDark);
        });
    }

    // Listen for system preference changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only auto-switch if user hasn't manually set preference
            if (!localStorage.getItem('theme')) {
                body.classList.toggle('dark-mode', e.matches);
                updateToggleUI(e.matches);
            }
        });
    }
});
