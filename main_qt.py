"""
EchoZero Qt GUI Entry Point

Launches the Qt-based graphical interface for EchoZero.
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation warning from librosa (librosa/core/intervals.py).
# Librosa still uses pkg_resources; remove this filter when librosa migrates to importlib.resources.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)

# When frozen (PyInstaller), bundle root must be on path so bundled packages (e.g. dotenv) are found
if getattr(sys, "frozen", False):
    _bundle_root = Path(sys.executable).parent
    if str(_bundle_root) not in sys.path:
        sys.path.insert(0, str(_bundle_root))
    # Windows: Add torch/lib to DLL search path so c10.dll can find its dependencies.
    # PyInstaller 6+ puts collected files in _internal/; without this, WinError 1114 occurs.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _internal = _bundle_root / "_internal"
        _torch_lib = _internal / "torch" / "lib"
        if _torch_lib.is_dir():
            try:
                os.add_dll_directory(str(_torch_lib))
            except OSError:
                pass  # Ignore if add_dll_directory fails (e.g. older Python)
# Add project root to path so src is importable (needed before loading .env from paths)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load environment variables (see docs/packaging/PACKAGING.md).
# Priority: bundled defaults -> app-local .env -> user-data .env (highest).
from dotenv import load_dotenv
try:
    from src.utils.paths import get_app_install_dir, get_user_data_dir
    _install_dir = get_app_install_dir()
    _user_dir = get_user_data_dir()
    _user_env = _user_dir / ".env"
    _bundled_example = _install_dir / ".env.example" if _install_dir else None
    _user_example = _user_dir / ".env.example"
    _runtime_dirs = []
    for _candidate in (
        _install_dir,
        Path(__file__).resolve().parent,
        (_install_dir / "_internal") if _install_dir else None,
    ):
        if _candidate and _candidate not in _runtime_dirs:
            _runtime_dirs.append(_candidate)

    # First run: copy bundled .env.example to user data dir so user has a template
    if _bundled_example and _bundled_example.exists() and not _user_example.exists():
        import shutil
        shutil.copy2(_bundled_example, _user_example)

    # Lowest priority: build-time bundled config (enables zero-config shipping)
    for _runtime_dir in _runtime_dirs:
        _bundled = _runtime_dir / "bundled_config.env"
        if _bundled.exists():
            load_dotenv(_bundled)
            break

    # App-local .env (if present)
    for _runtime_dir in _runtime_dirs:
        _runtime_env = _runtime_dir / ".env"
        if _runtime_env.exists():
            load_dotenv(_runtime_env, override=True)
            break

    # Highest priority: user-specific .env in Application Support
    if _user_env.exists():
        load_dotenv(_user_env, override=True)
except Exception:
    _dev_env = Path(__file__).resolve().parent / ".env"
    if _dev_env.exists():
        load_dotenv(_dev_env)

# CRITICAL: Set OpenBLAS/MKL threading to single-threaded BEFORE importing NumPy
# This prevents conflicts with Qt threading that cause crashes in NumPy operations
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Clear pycache if enabled in settings (prevents stale bytecode issues)
# This runs BEFORE other imports to ensure fresh bytecode is compiled
from src.utils.pycache_cleaner import clear_pycache_if_enabled
clear_pycache_if_enabled(verbose=False)

# Import Qt BEFORE other imports to ensure QApplication exists
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from src.application.bootstrap import initialize_services
from src.application.bootstrap_loading_progress import LoadingProgressTracker
from src.application.events import init_event_dispatcher
from ui.qt_gui.qt_application import QtEchoZeroApp
from ui.qt_gui.widgets.splash_screen import SplashScreen
from src.utils.message import Log


def _maybe_run_subprocess_cli() -> bool:
    """
    Run block-execution CLI mode when launched with --run-block-cli.

    In packaged builds, sys.executable is the app binary (not a Python interpreter),
    so the UI process cannot spawn `-m src.features.execution.run_block_cli`.
    Instead it relaunches the same executable with this flag and we dispatch here.
    """
    if "--run-block-cli" not in sys.argv:
        return False
    idx = sys.argv.index("--run-block-cli")
    sys.argv = [sys.argv[0]] + sys.argv[idx + 1 :]
    from src.features.execution.run_block_cli import main as run_block_cli_main
    raise SystemExit(run_block_cli_main())


def _validate_auth_config() -> tuple[str, str]:
    """
    Validate auth configuration before any auth flow.
    
    Single pathway: MEMBERSTACK_APP_SECRET is required. No fallbacks.
    
    Returns:
        Tuple of (app_secret, verify_url).
    
    Raises:
        SystemExit: If config is invalid. Shows clear error and solution before exiting.
    """
    app_secret = (os.getenv("MEMBERSTACK_APP_SECRET") or "").strip()
    verify_url = (os.getenv("MEMBERSTACK_VERIFY_URL") or "").strip() or "https://echozero-auth.speeoflight.workers.dev"
    
    if not app_secret:
        from src.utils.paths import get_user_data_dir
        from pathlib import Path
        user_env = get_user_data_dir() / ".env"
        project_env = Path(__file__).resolve().parent / ".env"
        Log.error("MEMBERSTACK_APP_SECRET is required but not set.")
        project_root = Path(__file__).resolve().parent
        msg = (
            "EchoZero requires MEMBERSTACK_APP_SECRET to authenticate.\n\n"
            "To fix:\n"
            "1. Copy .env.example to .env\n"
            "2. Set MEMBERSTACK_APP_SECRET=<your_secret> in that file\n"
            "3. Place .env in project root: " + str(project_root) + "\n"
            "   Or in: " + str(user_env) + "\n\n"
            "Get the secret from your EchoZero deployment admin, or set it at build time:\n"
            "MEMBERSTACK_APP_SECRET=... python scripts/build_app.py\n\n"
            "See docs/packaging/PACKAGING.md for details."
        )
        from PyQt6.QtWidgets import QMessageBox
        app = QApplication.instance()
        if app:
            QMessageBox.critical(None, "Auth Configuration Required", msg)
        else:
            print("ERROR: " + msg.replace("\n\n", "\n"))
        raise SystemExit(1)
    
    return app_secret, verify_url


def _create_auth_services():
    """
    Create and return the shared authentication service instances.
    
    Config must be valid (validated earlier). No fallbacks.
    
    Returns:
        Tuple of (MemberstackAuth, TokenStorage, LicenseLeaseManager).
    """
    from src.infrastructure.auth.memberstack_auth import MemberstackAuth
    from src.infrastructure.auth.token_storage import TokenStorage
    from src.infrastructure.auth.license_lease import LicenseLeaseManager
    
    app_secret, verify_url = _validate_auth_config()
    
    token_storage = TokenStorage()
    ms_auth = MemberstackAuth(verify_url=verify_url, app_secret=app_secret)
    lease_manager = LicenseLeaseManager(token_storage, app_secret=app_secret)
    
    return ms_auth, token_storage, lease_manager


def _authenticate(app: QApplication) -> bool:
    """
    Lease-aware authentication gate.
    
    Flow:
        1. Check for a valid stored lease (signed, not expired, clock OK)
        2. If lease is valid and a server check is due, try silent re-check
           - Online + active  -> renew lease, proceed
           - Online + expired -> revoke lease, show login dialog
           - Offline/error    -> proceed with existing lease (grace period)
        3. If lease is expired or missing -> show login dialog
        4. On successful login -> grant a lease anchored to billing period
    
    Returns:
        True if authenticated (with valid lease), False if user quit.
    """
    from src.infrastructure.auth.license_lease import LeaseStatus
    
    ms_auth, token_storage, lease_manager = _create_auth_services()
    
    # Store services on the module so the monitor can reuse them later
    global _auth_services
    _auth_services = (ms_auth, token_storage, lease_manager)
    
    # --- Step 1: Check existing lease ---
    lease_status = lease_manager.validate_lease()
    Log.info(f"Authentication: Lease status = {lease_status.value}")
    
    if lease_status == LeaseStatus.VALID:
        # Lease is valid -- try a silent server re-check if one is due
        if lease_manager.needs_server_check():
            Log.info("Authentication: Server re-check is due, attempting...")
            credentials = token_storage.get_credentials()
            member_id = credentials.get("member_id", "") if credentials else ""
            token = credentials.get("token", "") if credentials else ""
            
            if member_id:
                member_info = ms_auth.verify_member(member_id, token=token)
                if member_info:
                    # Server confirmed active -- renew the lease
                    lease_manager.renew_lease(
                        billing_period_end_at=member_info.get("billing_period_end", ""),
                    )
                    Log.info("Authentication: Lease renewed via server re-check.")
                    return True
                else:
                    # Hard failure mode: any failed re-check invalidates local auth.
                    Log.info(
                        f"Authentication: Re-check failed ({ms_auth.last_error_kind}). "
                        "Clearing lease and credentials."
                    )
                    lease_manager.revoke_lease()
                    token_storage.clear_credentials()
                    # Fall through to login dialog
            else:
                # No stored credentials but lease exists -- unusual, proceed
                Log.info("Authentication: Valid lease, no stored credentials for re-check.")
                return True
        else:
            # Lease valid and no re-check needed -- fast path
            days = lease_manager.get_days_remaining()
            Log.info(f"Authentication: Valid lease, {days} days remaining.")
            return True
    
    elif lease_status == LeaseStatus.CLOCK_TAMPERED:
        # Clock was set backward -- must verify online
        Log.warning("Authentication: Clock tampering detected, forcing server check.")
        credentials = token_storage.get_credentials()
        member_id = credentials.get("member_id", "") if credentials else ""
        token = credentials.get("token", "") if credentials else ""
        if member_id:
            member_info = ms_auth.verify_member(member_id, token=token)
            if member_info:
                lease_manager.renew_lease(
                    billing_period_end_at=member_info.get("billing_period_end", ""),
                )
                Log.info("Authentication: Clock issue resolved, lease renewed.")
                return True
        # If we can't verify, fall through to login
    
    # --- Step 2: Lease expired, invalid, or missing -- show login dialog ---
    if lease_status == LeaseStatus.EXPIRED:
        Log.info("Authentication: Lease expired. Online re-activation required.")
    elif lease_status == LeaseStatus.INVALID_SIGNATURE:
        Log.warning("Authentication: Lease signature invalid. Re-login required.")
        lease_manager.revoke_lease()
        token_storage.clear_credentials()
    elif lease_status == LeaseStatus.MISSING:
        Log.info("Authentication: No lease found. Login required.")
    
    return _show_login_and_grant_lease(ms_auth, token_storage, lease_manager)


def _show_login_and_grant_lease(ms_auth, token_storage, lease_manager) -> bool:
    """
    Show the login dialog. On success, grant a new lease.
    
    Returns:
        True if login succeeded and lease was granted.
    """
    from ui.qt_gui.dialogs.login_dialog import LoginDialog
    
    Log.info("Authentication: Showing login dialog.")
    login_dialog = LoginDialog(ms_auth, token_storage)
    result = login_dialog.exec()
    
    if result != LoginDialog.DialogCode.Accepted:
        return False
    
    # Login succeeded -- grant a lease from stored credentials
    credentials = token_storage.get_credentials()
    if credentials:
        member_id = credentials.get("member_id", "")
        member_data = credentials.get("member_data", {})
        email = member_data.get("email", "") if isinstance(member_data, dict) else ""
        lease_manager.grant_lease(
            member_id=member_id,
            email=email,
            billing_period_end_at=member_data.get("billing_period_end", "") if isinstance(member_data, dict) else "",
        )
    
    return True


# Module-level storage for auth services, set during _authenticate()
_auth_services = None


def _start_license_monitor(qt_app) -> None:
    """
    Create and start the background license monitor.
    
    Connects signals so that license expiration mid-session triggers
    the save-and-lock dialog.
    
    Args:
        qt_app: The running QtEchoZeroApp instance (for save/exit).
    """
    from src.infrastructure.auth.license_monitor import LicenseMonitor
    from ui.qt_gui.dialogs.license_expired_dialog import LicenseExpiredDialog
    
    global _auth_services, _license_monitor
    if _auth_services is None:
        Log.warning("LicenseMonitor: Auth services not initialized, skipping.")
        return
    
    ms_auth, token_storage, lease_manager = _auth_services
    
    monitor = LicenseMonitor(ms_auth, token_storage, lease_manager)
    _license_monitor = monitor  # prevent GC
    
    def on_license_expired(reason: str):
        """Handle license expiration during an active session."""
        Log.info(f"License expired mid-session: {reason}")
        
        # Stop the monitor while the dialog is shown
        monitor.stop()
        
        dialog = LicenseExpiredDialog(
            reason=reason,
            ms_auth=ms_auth,
            token_storage=token_storage,
            lease_manager=lease_manager,
            parent=qt_app.main_window if hasattr(qt_app, 'main_window') else None,
        )
        result = dialog.exec()
        
        if result == LicenseExpiredDialog.DialogCode.Accepted:
            # Reactivation succeeded -- resume the monitor
            Log.info("License reactivated. Resuming monitor.")
            monitor.start()
        else:
            # User chose "Save Work and Exit"
            Log.info("User chose to save and exit after license expiration.")
            try:
                if hasattr(qt_app, 'facade') and qt_app.facade:
                    qt_app.facade.save_project()
                    Log.info("Project saved before license exit.")
            except Exception as e:
                Log.error(f"Failed to save project on license exit: {e}")
            
            # Force application exit
            QApplication.instance().quit()
    
    def on_license_warning(days_remaining: int):
        """Log license warning (could show status bar indicator in the future)."""
        Log.warning(
            f"License lease expires in {days_remaining} days. "
            f"Connect to the internet to renew."
        )
    
    monitor.license_expired.connect(on_license_expired)
    monitor.license_warning.connect(on_license_warning)
    monitor.start()


# Module-level reference to prevent garbage collection of the monitor
_license_monitor = None


def main():
    """Main entry point for Qt GUI"""
    # Create QApplication FIRST (required for splash screen)
    app = QApplication(sys.argv)
    app.setApplicationName("EchoZero")
    app.setOrganizationName("EchoZero")
    
    # -- Authentication Gate --
    # User must be logged in before the app loads anything
    if not _authenticate(app):
        Log.info("User cancelled login. Exiting.")
        return 0
    
    # Create splash screen
    splash = SplashScreen()
    splash.show_and_process_events()
    
    # Create progress tracker with splash screen as callback
    progress_tracker = LoadingProgressTracker(callback=splash)
    
    # Timer to update overall progress during loading
    def update_overall_progress():
        """Update overall progress bar from tracker"""
        progress = progress_tracker.get_overall_progress()
        splash.update_overall_progress(progress)
        app.processEvents()  # Keep UI responsive
    
    progress_timer = QTimer()
    progress_timer.timeout.connect(update_overall_progress)
    progress_timer.start(50)  # Update every 50ms
    
    Log.info("=" * 60)
    Log.info("EchoZero Qt GUI")
    Log.info("=" * 60)
    
    try:
        # Initialize core services with progress tracking
        Log.info("Initializing EchoZero services...")
        container = initialize_services(progress_tracker=progress_tracker)
        
        # Stop progress timer
        progress_timer.stop()
        
        # Final progress update
        splash.update_overall_progress(1.0)
        app.processEvents()
        
        # Create and initialize Qt application
        Log.info("Initializing Qt GUI...")
        qt_app = QtEchoZeroApp()
        
        # Initialize the event dispatcher on the main thread (before any background work)
        init_event_dispatcher()
        
        # Initialize Qt GUI with progress tracking
        qt_app.initialize(container.facade, progress_tracker=progress_tracker)
        
        # -- Start License Monitor --
        # Background re-validation of the license lease while the app runs
        _start_license_monitor(qt_app)
        
        # Close splash screen after a brief delay to show completion
        QTimer.singleShot(300, splash.close)
        
        # Run the application
        Log.info("Starting application...")
        exit_code = qt_app.run()
        
        # Cleanup
        Log.info("Shutting down...")
        
        # Stop license monitor
        global _license_monitor
        if _license_monitor:
            _license_monitor.stop()
        
        qt_app.shutdown()
        
        # Cleanup service container resources (database, sockets, threads)
        if container:
            container.cleanup()
        
        Log.info("EchoZero Qt GUI exited successfully")
        return exit_code
        
    except Exception as e:
        Log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        _write_crash_log(e)
        
        # Show error in splash screen before closing
        if splash.isVisible():
            splash.step_label.setText(f"Error: {str(e)}")
            splash.step_label.setStyleSheet("color: #000000;")
            app.processEvents()
            QTimer.singleShot(2000, splash.close)
            app.processEvents()
        
        return 1


if __name__ == "__main__":
    _maybe_run_subprocess_cli()
    sys.exit(main())

