# Settings System

Centralized settings management with persistence.

## Overview

The settings system provides:
- Type-safe settings access
- Persistence to database
- Category-based organization
- UI integration support

## Architecture

```
settings/
├── base_settings.py      # Base settings class
├── settings_service.py   # Settings operations
├── settings_provider.py  # Settings access
└── [feature]_settings.py # Feature-specific settings
```

## Creating Settings

```python
from src.application.settings.base_settings import BaseSettings

class MyFeatureSettings(BaseSettings):
    CATEGORY = "my_feature"
    
    @property
    def my_option(self) -> str:
        return self.get("my_option", "default_value")
    
    @my_option.setter
    def my_option(self, value: str):
        self.set("my_option", value)
```

## Usage

```python
from src.application.settings.my_feature_settings import MyFeatureSettings

settings = MyFeatureSettings()
print(settings.my_option)
settings.my_option = "new_value"
```

## Related

- [Encyclopedia: Settings](../../../docs/encyclopedia/)
