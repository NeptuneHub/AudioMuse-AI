document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');

    // The menu is now positioned off-screen by default via CSS.
    // This script handles the open/close classes plus focus/ARIA state so the
    // off-screen sidebar isn't tabbable and screen readers know its state.

    const focusFirstInSidebar = () => {
        const first = sidebar.querySelector('a, button');
        if (first) first.focus();
    };

    // Keep the toggle's aria-expanded in sync and make the off-screen sidebar
    // unreachable (inert removes it from tab order + a11y tree; aria-hidden is a
    // fallback for engines without inert support).
    const setSidebarA11y = (isOpen) => {
        if (menuToggle) menuToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        if (isOpen) {
            sidebar.removeAttribute('inert');
            sidebar.removeAttribute('aria-hidden');
        } else {
            sidebar.setAttribute('inert', '');
            sidebar.setAttribute('aria-hidden', 'true');
        }
    };

    // Fixed elements attach to the LAYOUT viewport: under pinch-zoom the
    // visual viewport pans away from it and the open menu's bottom edge
    // appears mid-screen. While open, pin the sidebar to the visual viewport.
    const syncSidebarToVisualViewport = () => {
        if (!window.visualViewport) return;
        if (!sidebar.classList.contains('open')) {
            sidebar.style.top = '';
            sidebar.style.left = '';
            sidebar.style.height = '';
            return;
        }
        const vv = window.visualViewport;
        sidebar.style.top = vv.offsetTop + 'px';
        sidebar.style.left = vv.offsetLeft + 'px';
        sidebar.style.height = vv.height + 'px';
    };
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', syncSidebarToVisualViewport);
        window.visualViewport.addEventListener('scroll', syncSidebarToVisualViewport);
    }

    // Function to open the menu
    const openMenu = ({ focus = true } = {}) => {
        sidebar.classList.add('open');
        mainContent.classList.add('sidebar-open');
        document.documentElement.classList.add('sidebar-open');
        localStorage.setItem('menuOpen', 'true');
        setSidebarA11y(true);
        syncSidebarToVisualViewport();
        if (focus) focusFirstInSidebar();
    };

    // Function to close the menu
    const closeMenu = ({ returnFocus = false } = {}) => {
        sidebar.classList.remove('open');
        mainContent.classList.remove('sidebar-open');
        document.documentElement.classList.remove('sidebar-open');
        localStorage.setItem('menuOpen', 'false');
        setSidebarA11y(false);
        syncSidebarToVisualViewport();
        if (returnFocus && menuToggle) menuToggle.focus();
    };

    // Sync classes if menu was opened by FOUC prevention script (don't steal
    // focus on load); otherwise mark the closed sidebar inert.
    if (document.documentElement.classList.contains('sidebar-open')) {
        sidebar.classList.add('open');
        mainContent.classList.add('sidebar-open');
        setSidebarA11y(true);
        syncSidebarToVisualViewport();
    } else {
        setSidebarA11y(false);
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

    // Escape closes the menu and returns focus to the toggle
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && sidebar.classList.contains('open')) {
            closeMenu({ returnFocus: true });
        }
    });

    // --- Submenu accordion toggle ---
    document.querySelectorAll('.has-submenu > .submenu-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const parent = toggle.closest('.has-submenu');
            const isOpen = parent.classList.toggle('open');
            toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        });
    });

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
