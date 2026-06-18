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
        document.documentElement.classList.add('sidebar-open');
        localStorage.setItem('menuOpen', 'true');
    };

    // Function to close the menu
    const closeMenu = () => {
        sidebar.classList.remove('open');
        mainContent.classList.remove('sidebar-open');
        document.documentElement.classList.remove('sidebar-open');
        localStorage.setItem('menuOpen', 'false');
    };

    // Sync classes if menu was opened by FOUC prevention script
    if (document.documentElement.classList.contains('sidebar-open')) {
        sidebar.classList.add('open');
        mainContent.classList.add('sidebar-open');
    }

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

    // --- Submenu accordion toggle ---
    document.querySelectorAll('.has-submenu > .submenu-toggle').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            link.closest('.has-submenu').classList.toggle('open');
        });
    });

    // Donate button
    const donateLink = document.createElement('a');
    donateLink.className = 'donate-button';
    donateLink.href = 'https://liberapay.com/NeptuneHub/donate';
    donateLink.target = '_blank';
    donateLink.rel = 'noopener noreferrer';
    donateLink.setAttribute('aria-label', 'Donate to AudioMuse-AI');
    donateLink.innerHTML = '<svg class="donate-heart" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg><span>Donate</span>';
    sidebar.appendChild(donateLink);

    // Display App Version from meta tag
    const versionMeta = document.querySelector('meta[name="app-version"]');
    if (versionMeta && versionMeta.content) {
        const appVersion = versionMeta.content;
        const versionElement = document.createElement('div');
        versionElement.className = 'app-version'; // For styling
        versionElement.textContent = `AudioMuse-AI - ${appVersion}`;
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
            e.stopPropagation();
            body.classList.toggle('dark-mode');
            document.documentElement.classList.toggle('dark-mode');
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
