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
});
