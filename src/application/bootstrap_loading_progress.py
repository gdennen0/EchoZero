"""
Loading Progress Tracking

Provides interface for tracking application loading progress with module and sub-component granularity.
"""
from typing import Protocol, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class LoadingStep:
    """Represents a single loading step within a module"""
    name: str
    description: Optional[str] = None


@dataclass
class LoadingModule:
    """Represents a module being loaded"""
    name: str
    description: Optional[str] = None
    total_steps: int = 1
    current_step: int = 0
    current_step_name: Optional[str] = None


class LoadingProgressCallback(Protocol):
    """Protocol for progress callbacks"""
    
    def on_module_start(self, module_name: str, total_steps: int = 1) -> None:
        """Called when a module starts loading"""
        ...
    
    def on_module_step(self, module_name: str, step_name: str, step_number: int, total_steps: int) -> None:
        """Called when a step within a module completes"""
        ...
    
    def on_module_complete(self, module_name: str) -> None:
        """Called when a module finishes loading"""
        ...
    
    def on_error(self, module_name: str, error: Exception) -> None:
        """Called when a module fails to load"""
        ...


class LoadingProgressTracker:
    """
    Tracks loading progress across multiple modules.
    
    Provides simple interface for reporting progress that can be used
    with or without a UI (splash screen).
    """
    
    def __init__(self, callback: Optional[LoadingProgressCallback] = None):
        """
        Initialize progress tracker.
        
        Args:
            callback: Optional callback for progress updates
        """
        self.callback = callback
        self.modules: List[LoadingModule] = []
        self.current_module_index = -1
        self.total_modules = 0
    
    def set_total_modules(self, count: int) -> None:
        """Set the total number of modules to load"""
        self.total_modules = count
    
    def start_module(self, name: str, description: Optional[str] = None, total_steps: int = 1) -> None:
        """
        Start loading a new module.
        
        Args:
            name: Module name
            description: Optional description
            total_steps: Number of steps in this module
        """
        module = LoadingModule(
            name=name,
            description=description,
            total_steps=total_steps,
            current_step=0
        )
        self.modules.append(module)
        self.current_module_index = len(self.modules) - 1
        
        if self.callback:
            self.callback.on_module_start(name, total_steps)
    
    def update_step(self, step_name: str, step_number: Optional[int] = None) -> None:
        """
        Update progress within current module.
        
        Args:
            step_name: Name of the current step
            step_number: Optional step number (auto-increments if not provided)
        """
        if self.current_module_index < 0 or self.current_module_index >= len(self.modules):
            return
        
        module = self.modules[self.current_module_index]
        
        if step_number is None:
            module.current_step += 1
        else:
            module.current_step = step_number
        
        module.current_step_name = step_name
        
        if self.callback:
            self.callback.on_module_step(
                module.name,
                step_name,
                module.current_step,
                module.total_steps
            )
    
    def complete_module(self) -> None:
        """Mark current module as complete"""
        if self.current_module_index < 0 or self.current_module_index >= len(self.modules):
            return
        
        module = self.modules[self.current_module_index]
        module.current_step = module.total_steps
        
        if self.callback:
            self.callback.on_module_complete(module.name)
    
    def report_error(self, error: Exception) -> None:
        """Report an error in the current module"""
        if self.current_module_index < 0 or self.current_module_index >= len(self.modules):
            return
        
        module = self.modules[self.current_module_index]
        
        if self.callback:
            self.callback.on_error(module.name, error)
    
    def get_overall_progress(self) -> float:
        """
        Get overall progress as a float between 0.0 and 1.0.
        
        Returns:
            Progress value (0.0 = not started, 1.0 = complete)
        """
        if self.total_modules == 0:
            return 0.0
        
        completed = 0.0
        for module in self.modules:
            if module.current_step >= module.total_steps:
                completed += 1.0
            else:
                completed += module.current_step / module.total_steps
        
        return completed / self.total_modules
    
    def get_current_module(self) -> Optional[LoadingModule]:
        """Get the currently loading module"""
        if self.current_module_index >= 0 and self.current_module_index < len(self.modules):
            return self.modules[self.current_module_index]
        return None
