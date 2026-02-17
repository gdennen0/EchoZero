"""
Shared module containing cross-cutting concerns.

This module contains code that is used across multiple features.
Organized by layer following the vertical module pattern.

Structure:
    shared/
        application/
            events/         - EventBus and domain events
            registry/       - ModuleRegistry for component registration
            settings/       - SettingsRegistry for settings auto-discovery
            status/         - StatusPublisher for unified status updates
            validation/     - Validation framework for inputs/forms
        domain/
            entities/       - Shared entities (DataItem, etc.)
        infrastructure/
            persistence/    - BaseRepository pattern
        utils/              - Logging, paths, and other utilities

Usage:
    # Events
    from src.shared.application.events import EventBus, BlockAdded
    
    # Settings
    from src.shared.application.settings import SettingsRegistry, register_block_settings
    
    # Status
    from src.shared.application.status import StatusPublisher, StatusLevel
    
    # Registry
    from src.shared.application.registry import ModuleRegistry, register_component
    
    # Validation
    from src.shared.application.validation import validate, RequiredValidator
    
    # Repository
    from src.shared.infrastructure.persistence import BaseRepository
    
    # Entities
    from src.shared.domain.entities import DataItem, AudioDataItem
    
    # Utils
    from src.shared.utils import Log, get_user_data_dir
"""
