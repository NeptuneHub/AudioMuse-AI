document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const currentPath = window.location.pathname;

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

    // Define menu items
    const menuItems = [
        { href: '/', text: 'Analysis and Clustering' },
        { href: '/chat', text: 'Instant Playlist' },
        { href: '/similarity', text: 'Playlist from Similar Song' },
        //{ href: '/music_map', text: 'Music Map (t-SNE)' },
        { href: '/path', text: 'Song Path' },
        { href: '/alchemy', text: 'Song Alchemy' },
        { href: '/sonic_fingerprint', text: 'Sonic Fingerprint' },
        //{ href: '/collection', text: 'Collection Sync' }, // New Menu Item - commented because still experimental
        { href: '/cleaning', text: 'Cleaning' }
    ];

    // Find the navigation list
    const navList = document.querySelector('.sidebar-nav');

    // Generate menu items
    menuItems.forEach(item => {
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.href = item.href;
        link.textContent = item.text;

        // Check for active page
        const linkPath = item.href;
        let isActive = false;
        if (linkPath === '/') {
            if (currentPath === '/' || currentPath.endsWith('/index.html')) {
                isActive = true;
            }
        } else {
            // Check if the current path starts with the link's path.
            // This handles cases like '/sonic_fingerprint' and '/sonic_fingerprint/'
            if (currentPath.startsWith(linkPath) && (currentPath.length === linkPath.length || currentPath.charAt(linkPath.length) === '/')) {
                isActive = true;
            }
        }

        if (isActive) {
            link.classList.add('active');
        }

        // Add event listener to auto-close menu on link click
        link.addEventListener('click', (e) => {
            if (!sidebar.classList.contains('open')) {
                return; 
            }
            e.preventDefault();
            closeMenu();
            if (!isActive) {
                setTimeout(() => {
                    window.location.href = link.href;
                }, 300); // Match CSS transition
            }
        });

        listItem.appendChild(link);
        navList.appendChild(listItem);
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
