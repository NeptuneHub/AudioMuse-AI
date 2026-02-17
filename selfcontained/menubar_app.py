"""
macOS Menu Bar App for AudioMuse-AI
Shows a menu bar icon with quick access to logs, config, and web interface
"""
import os
import sys
import subprocess
import webbrowser
import threading
from pathlib import Path

try:
    import rumps
    RUMPS_AVAILABLE = True
except ImportError:
    RUMPS_AVAILABLE = False
    print("rumps not available - menu bar icon disabled")


class AudioMuseMenuBarApp(rumps.App):
    def __init__(self, server_url='http://localhost:8000'):
        super(AudioMuseMenuBarApp, self).__init__(
            "AudioMuse-AI",
            icon=None,  # Will use default music note or custom icon
            quit_button=None  # We'll create custom quit
        )
        
        self.server_url = server_url
        self.data_dir = Path.home() / '.audiomuse'
        self.log_file = self.data_dir / 'audiomuse.log'
        self.config_file = self.data_dir / 'config.ini'
        
        # Create menu items
        self.menu = [
            rumps.MenuItem('🌐 Open Web Interface', callback=self.open_browser),
            rumps.separator,
            rumps.MenuItem('📋 View Logs', callback=self.open_logs),
            rumps.MenuItem('⚙️ Open Configuration', callback=self.open_config),
            rumps.MenuItem('📁 Open Data Folder', callback=self.open_data_folder),
            rumps.separator,
            rumps.MenuItem('Server Status: Starting...', callback=None),
            rumps.separator,
            rumps.MenuItem('❌ Quit AudioMuse-AI', callback=self.quit_app),
        ]
        
        self.status_item = self.menu['Server Status: Starting...']
        
    def update_status(self, status_text):
        """Update the status menu item"""
        if self.status_item:
            self.status_item.title = f'Server Status: {status_text}'
    
    def open_browser(self, _):
        """Open web interface in browser"""
        webbrowser.open(self.server_url)
    
    def open_logs(self, _):
        """Open log file in Console.app"""
        if self.log_file.exists():
            subprocess.run(['open', '-a', 'Console', str(self.log_file)])
        else:
            rumps.alert(
                'Log File Not Found',
                f'Log file does not exist yet: {self.log_file}'
            )
    
    def open_config(self, _):
        """Open the web setup page (preferred) instead of editing the raw file."""
        try:
            setup_url = self.server_url.rstrip('/') + '/setup'
            webbrowser.open(setup_url)
        except Exception:
            # Fallback: open the raw config file if browser/open fails
            if self.config_file.exists():
                subprocess.run(['open', str(self.config_file)])
            else:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                self.config_file.write_text(
                    "# AudioMuse-AI Configuration\n"
                    "# Created by AudioMuse-AI\n"
                )
                subprocess.run(['open', str(self.config_file)])
    
    def open_data_folder(self, _):
        """Open data folder in Finder"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['open', str(self.data_dir)])
    
    def quit_app(self, _):
        """Quit the application safely.

        Uses rumps.quit_application() for a normal shutdown, raises
        SystemExit to allow Python-level cleanup, and schedules a
        short-timer hard exit (os._exit) as a failsafe to avoid
        event-loop hangs or deallocation races with AppKit.
        """
        # Best-effort graceful quit via rumps
        try:
            rumps.quit_application()
        except Exception as e:
            # If rumps fails, proceed to a controlled shutdown
            try:
                import logging
                logging.getLogger(__name__).warning(f"rumps.quit_application() raised: {e}")
            except Exception:
                pass

        # Failsafe: force immediate process exit shortly after to avoid hangs
        try:
            threading.Timer(0.2, os._exit, args=(0,)).start()
        except Exception:
            pass

        # Raise SystemExit to allow proper Python cleanup (handlers, atexit, etc.)
        sys.exit(0)


def run_menubar_app(server_url='http://localhost:8000', on_ready_callback=None):
    """
    Run the menu bar app
    
    Args:
        server_url: URL of the web interface
        on_ready_callback: Function to call when menu bar is ready
    """
    if not RUMPS_AVAILABLE:
        print("Menu bar app not available (rumps not installed)")
        if on_ready_callback:
            on_ready_callback()
        return None
    
    app = AudioMuseMenuBarApp(server_url)
    
    # Call the ready callback in a thread so it doesn't block the menu bar
    if on_ready_callback:
        def run_callback():
            import time
            time.sleep(0.5)  # Give menu bar time to initialize
            on_ready_callback()
        
        callback_thread = threading.Thread(target=run_callback, daemon=True)
        callback_thread.start()
    
    return app
