"""
Plot Events Block - Universal Visualization System

Modular visualization block for EchoZero data types.
Supports EventDataItem with proper axis scaling and extensible architecture.
"""
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import EventDataItem, Event
from src.application.processing.block_processor import BlockProcessor
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import (
    track_progress, get_progress_tracker
)
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    style: str = "timeline"
    theme: str = "light"
    width: float = 14.0
    height: float = 7.0
    dpi: int = 200
    output_dir: Path = Path("data/visualizations")
    show_labels: bool = True
    show_grid: bool = True
    color_by: str = "classification"
    auto_scale: bool = True
    
    @classmethod
    def from_block(cls, block: Block) -> 'PlotConfig':
        """Create config from block metadata."""
        # Output directory
        output_dir = Path(block.metadata.get("output_dir", "data/visualizations"))
        
        # Backward compatibility: map old style names
        style = block.metadata.get("plot_style", "timeline")
        style_map = {"bars": "timeline", "markers": "markers"}
        style = style_map.get(style, style)
        
        return cls(
            style=style,
            theme=block.metadata.get("theme", "light"),
            width=float(block.metadata.get("figsize_width", 14.0)),
            height=float(block.metadata.get("figsize_height", 7.0)),
            dpi=int(block.metadata.get("dpi", 200)),
            output_dir=output_dir,
            show_labels=block.metadata.get("show_labels", True),
            show_grid=block.metadata.get("show_grid", True),
            color_by=block.metadata.get("color_by", "classification"),
            auto_scale=block.metadata.get("auto_scale", True),
        )


# =============================================================================
# Style System
# =============================================================================

def get_style(theme: str) -> Dict[str, Any]:
    """Get style configuration for theme."""
    if theme == "dark":
        return {
            "bg_color": "#1a1a1a",
            "fg_color": "#ffffff",
            "grid_color": "#333333",
            "accent_colors": ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52BE80'],
            "alpha": 0.85,
        }
    else:  # light
        return {
            "bg_color": "#ffffff",
            "fg_color": "#2c3e50",
            "grid_color": "#e0e0e0",
            "accent_colors": ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                            '#1ABC9C', '#E67E22', '#34495E', '#16A085', '#27AE60'],
            "alpha": 0.9,
        }


# =============================================================================
# Data Adapters
# =============================================================================

class DataAdapter(ABC):
    """Abstract base for data type adapters."""
    
    @abstractmethod
    def can_handle(self, data_item: DataItem) -> bool:
        """Check if this adapter can handle the data item."""
        pass
    
    @abstractmethod
    def extract(self, data_item: DataItem) -> Dict[str, Any]:
        """Extract visualization data from data item."""
        pass


class EventDataAdapter(DataAdapter):
    """Adapter for EventDataItem."""
    
    def can_handle(self, data_item: DataItem) -> bool:
        return isinstance(data_item, EventDataItem)
    
    def extract(self, data_item: EventDataItem) -> Dict[str, Any]:
        """Extract event data for visualization."""
        events = data_item.get_events()
        
        if not events:
            return {
                "times": [],
                "durations": [],
                "classifications": [],
                "metadata": [],
                "type": "events",
            }
        
        # Extract all data
        times = [e.time for e in events]
        durations = [e.duration for e in events]
        classifications = [e.classification for e in events]
        metadata = [e.metadata for e in events]
        
        return {
            "times": times,
            "durations": durations,
            "classifications": classifications,
            "metadata": metadata,
            "type": "events",
            "events": events,  # Keep events for renderers
        }


# =============================================================================
# Axis Calculation (CRITICAL FIX)
# =============================================================================

def calculate_axes_limits(data: Dict[str, Any], padding: float = 0.05) -> Dict[str, float]:
    """
    Calculate proper axis limits from data.
    
    This is the critical fix - ensures axes scale correctly to actual data.
    """
    times = data.get("times", [])
    durations = data.get("durations", [])
    
    if not times:
        return {
            "x_min": 0.0,
            "x_max": 1.0,
            "x_range": 1.0,
        }
    
    # Calculate actual time range from data
    min_time = min(times)
    max_time = max(t + d for t, d in zip(times, durations)) if durations else max(times)
    time_range = max_time - min_time
    
    # Ensure minimum range (handle edge case where all events are at same time)
    if time_range < 0.01:
        time_range = 1.0
        max_time = min_time + time_range
    
    # Add padding
    padding_amount = max(0.1, time_range * padding)
    
    return {
        "x_min": max(0.0, min_time - padding_amount),
        "x_max": max_time + padding_amount,
        "x_range": time_range,
    }


# =============================================================================
# Renderers
# =============================================================================

def render_timeline(ax, data: Dict[str, Any], config: PlotConfig, style: Dict[str, Any], mpatches, np):
    """Render timeline visualization (horizontal bars)."""
    events = data.get("events", [])
    if not events:
        return
    
    # Get color map
    color_map = create_color_map(events, config.color_by, style)
    
    for i, event in enumerate(events):
        color = color_map.get(event.classification, style["accent_colors"][0]) if color_map else style["accent_colors"][0]
        
        if event.duration > 0:
            rect = mpatches.Rectangle(
                (event.time, i - 0.4), event.duration, 0.8,
                facecolor=color, alpha=style["alpha"],
                edgecolor=style["fg_color"], linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        else:
            ax.plot(event.time, i, 'o', color=color, markersize=10,
                   markeredgecolor=style["fg_color"], markeredgewidth=1.5, zorder=3)
        
        if config.show_labels and event.classification:
            label = event.classification
            if 'frequency_hz' in event.metadata:
                label += f"\n{event.metadata['frequency_hz']:.0f}Hz"
            ax.text(event.time + (event.duration / 2 if event.duration > 0 else 0), i,
                   label, fontsize=9, color=style["fg_color"],
                   ha='center', va='center', weight='medium',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=style["bg_color"],
                           alpha=0.8, edgecolor='none'))
    
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(events))],
                      color=style["fg_color"], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylabel("Event Index", color=style["fg_color"], fontsize=11, weight='bold')


def render_piano_roll(ax, data: Dict[str, Any], config: PlotConfig, style: Dict[str, Any], mpatches, np):
    """Render piano roll visualization."""
    events = data.get("events", [])
    if not events:
        return
    
    color_map = create_color_map(events, config.color_by, style)
    
    for event in events:
        midi_note = event.metadata.get('midi_note')
        if midi_note is None:
            midi_note = note_to_midi(event.classification) if event.classification else 60
        
        color = color_map.get(event.classification, style["accent_colors"][0]) if color_map else style["accent_colors"][0]
        
        if event.duration > 0:
            rect = mpatches.Rectangle(
                (event.time, midi_note - 0.4), event.duration, 0.8,
                facecolor=color, alpha=style["alpha"],
                edgecolor=style["fg_color"], linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        else:
            ax.plot(event.time, midi_note, 'o', color=color, markersize=10,
                   markeredgecolor=style["fg_color"], markeredgewidth=1.5, zorder=3)
    
    if events:
        midi_notes = []
        for e in events:
            note = e.metadata.get('midi_note')
            if note is None:
                note = note_to_midi(e.classification) if e.classification else 60
            midi_notes.append(note)
        
        min_note = max(0, min(midi_notes) - 2)
        max_note = min(127, max(midi_notes) + 2)
        ax.set_ylim(min_note, max_note)
        
        note_ticks = range(int(min_note), int(max_note) + 1, 2)
        note_labels = [midi_to_note(n) for n in note_ticks]
        ax.set_yticks(note_ticks)
        ax.set_yticklabels(note_labels, color=style["fg_color"], fontsize=9)
    
    ax.set_ylabel("Pitch (MIDI)", color=style["fg_color"], fontsize=11, weight='bold')


def render_markers(ax, data: Dict[str, Any], config: PlotConfig, style: Dict[str, Any], mpatches, np):
    """Render markers visualization."""
    events = data.get("events", [])
    if not events:
        return
    
    times = [e.time for e in events]
    color_map = create_color_map(events, config.color_by, style)
    colors = [color_map.get(e.classification, style["accent_colors"][0]) if color_map 
             else style["accent_colors"][0] for e in events]
    
    # Plot stems
    ax.stem(times, [1]*len(times), linefmt=style["grid_color"], markerfmt=' ',
           basefmt=' ', bottom=0, linewidths=1, alpha=0.5)
    
    # Plot markers
    for time, event, color in zip(times, events, colors):
        ax.plot(time, 1, 'o', color=color, markersize=12,
               markeredgecolor=style["fg_color"], markeredgewidth=1.5,
               zorder=3, alpha=style["alpha"])
        
        if event.duration > 0:
            ax.plot([time, time + event.duration], [0.95, 0.95],
                   color=color, linewidth=3, alpha=style["alpha"] * 0.7, zorder=2)
        
        if config.show_labels and event.classification:
            ax.text(time, 1.15, event.classification, fontsize=9,
                   color=style["fg_color"], rotation=45, ha='left', va='bottom',
                   weight='medium')
    
    ax.set_ylim(-0.1, 1.6)
    ax.set_yticks([])
    ax.set_ylabel('', color=style["fg_color"])


def render_scatter(ax, data: Dict[str, Any], config: PlotConfig, style: Dict[str, Any], mpatches, np):
    """Render scatter plot visualization."""
    events = data.get("events", [])
    if not events:
        return
    
    times = [e.time for e in events]
    durations = [e.duration for e in events]
    color_map = create_color_map(events, config.color_by, style)
    colors = [color_map.get(e.classification, style["accent_colors"][0]) if color_map 
             else style["accent_colors"][0] for e in events]
    
    sizes = [max(20, min(200, d * 50)) if d > 0 else 50 for d in durations]
    
    ax.scatter(times, durations, c=colors, s=sizes, alpha=style["alpha"],
              edgecolors=style["fg_color"], linewidths=1, zorder=3)
    
    if config.show_labels:
        for event, time, dur in zip(events, times, durations):
            if event.classification:
                ax.text(time, dur, event.classification, fontsize=8,
                       color=style["fg_color"], ha='center', va='bottom')
    
    ax.set_ylabel("Duration (s)", color=style["fg_color"], fontsize=11, weight='bold')


# Renderer registry
RENDERERS: Dict[str, Callable] = {
    "timeline": render_timeline,
    "piano_roll": render_piano_roll,
    "markers": render_markers,
    "scatter": render_scatter,
}

# Adapter registry
ADAPTERS: List[DataAdapter] = [
    EventDataAdapter(),
]


# =============================================================================
# Helper Functions
# =============================================================================

def create_color_map(events: List[Event], color_by: str, style: Dict[str, Any]) -> Dict[str, str]:
    """Create color mapping for events."""
    if color_by == "classification":
        classifications = list(set(e.classification for e in events if e.classification))
        if not classifications:
            return {}
        colors = style["accent_colors"] * ((len(classifications) // len(style["accent_colors"])) + 1)
        return {cls: colors[i] for i, cls in enumerate(classifications)}
    return {}


def note_to_midi(note_name: str) -> int:
    """Convert note name to MIDI number."""
    note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
               'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    if len(note_name) >= 2:
        if note_name[1] == '#':
            note = note_name[:2]
            octave = int(note_name[2:]) if len(note_name) > 2 else 4
        else:
            note = note_name[0]
            octave = int(note_name[1:]) if len(note_name) > 1 else 4
        return (octave + 1) * 12 + note_map.get(note, 0)
    return 60


def midi_to_note(midi: int) -> str:
    """Convert MIDI number to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


def apply_style(ax, style: Dict[str, Any]):
    """Apply styling to axes."""
    ax.set_facecolor(style["bg_color"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(style["grid_color"])
    ax.spines['bottom'].set_color(style["grid_color"])
    ax.tick_params(colors=style["fg_color"], labelsize=10)
    ax.xaxis.label.set_color(style["fg_color"])
    ax.yaxis.label.set_color(style["fg_color"])


# =============================================================================
# Main Processor
# =============================================================================

class PlotEventsProcessor(BlockProcessor):
    """Universal visualization processor for EchoZero data."""
    
    def get_block_type(self) -> str:
        return "PlotEvents"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for PlotEvents block.
        
        Status levels:
        - Warning (0): No inputs connected
        - Ready (1): Inputs connected or can plot
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_has_inputs(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if any inputs are connected."""
            if not hasattr(f, 'connection_service'):
                return True  # Can't check, assume ready
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id]
            return len(incoming) > 0
        
        return [
            BlockStatusLevel(
                priority=0,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_has_inputs]
            ),
            BlockStatusLevel(
                priority=1,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def can_process(self, block: Block) -> bool:
        return block.type == "PlotEvents"
    
    def process(self, block: Block, inputs: Dict[str, DataItem], metadata: Dict = None) -> Dict[str, DataItem]:
        """Process data and create visualization."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            raise RuntimeError("matplotlib not installed. Install with: pip install matplotlib")
        
        if not inputs or "events" not in inputs:
            Log.warning(f"PlotEvents: No events input for block {block.name}")
            return {}
        
        # Get configuration
        config = PlotConfig.from_block(block)
        style = get_style(config.theme)
        
        # Get progress tracker from metadata
        progress_tracker = get_progress_tracker(metadata)
        
        # Normalize input
        event_items = self._normalize_input(inputs["events"])
        
        # Process each event item with progress tracking
        for event_item in track_progress(event_items, progress_tracker, "Generating plots"):
            if not isinstance(event_item, EventDataItem):
                continue
            
            events = event_item.get_events()
            if not events:
                Log.warning(f"PlotEvents: No events found in {event_item.name}")
                continue
            
            self._create_plot(event_item, events, config, style, plt, mpatches, np)
        
        return {}
    
    def _normalize_input(self, events_input) -> List[EventDataItem]:
        """Normalize input to list of EventDataItem."""
        if isinstance(events_input, EventDataItem):
            return [events_input]
        elif isinstance(events_input, list):
            return [item for item in events_input if isinstance(item, EventDataItem)]
        return []
    
    def _create_plot(self, event_item: EventDataItem, events: List[Event],
                    config: PlotConfig, style: Dict[str, Any],
                    plt, mpatches, np):
        """Create visualization plot."""
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.time)
        
        # Find adapter
        adapter = None
        for a in ADAPTERS:
            if a.can_handle(event_item):
                adapter = a
                break
        
        if not adapter:
            Log.warning(f"PlotEvents: No adapter found for {event_item.type}")
            return
        
        # Extract data
        data = adapter.extract(event_item)
        data["events"] = sorted_events  # Ensure sorted events
        
        # Calculate axis limits (CRITICAL FIX)
        axes_limits = calculate_axes_limits(data)
        
        # Get renderer
        renderer = RENDERERS.get(config.style, render_timeline)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(config.width, config.height), dpi=config.dpi)
        apply_style(ax, style)
        
        # Render
        renderer(ax, data, config, style, mpatches, np)
        
        # Set axis limits (CRITICAL - from calculated data, not assumptions)
        ax.set_xlim(axes_limits["x_min"], axes_limits["x_max"])
        ax.set_xlabel("Time (seconds)", color=style["fg_color"], fontsize=11, weight='bold')
        
        # Title
        title = f"{event_item.name} | {len(events)} events"
        if event_item.metadata.get("source_audio_name"):
            title += f" | Source: {event_item.metadata['source_audio_name']}"
        ax.set_title(title, color=style["fg_color"], fontsize=13, weight='bold', pad=15)
        
        # Grid
        if config.show_grid:
            ax.grid(True, color=style["grid_color"], linestyle='--',
                   linewidth=0.5, alpha=0.5, zorder=0)
        
        # Legend
        if config.color_by == "classification" and sorted_events:
            classifications = list(set(e.classification for e in sorted_events if e.classification))
            if classifications and len(classifications) <= 15:
                color_map = create_color_map(sorted_events, config.color_by, style)
                handles = [mpatches.Patch(color=color_map.get(c, style["accent_colors"][0]),
                                         label=c, alpha=style["alpha"])
                          for c in sorted(classifications)]
                ax.legend(handles=handles, loc='upper right',
                         framealpha=0.9, facecolor=style["bg_color"],
                         edgecolor=style["grid_color"], fontsize=9,
                         labelcolor=style["fg_color"])
        
        # Save
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = config.output_dir / f"{event_item.name}_viz.png"
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight',
                   facecolor=style["bg_color"], edgecolor='none')
        plt.close(fig)
        
        Log.info(f"PlotEvents: Saved visualization to {output_path}")


# Auto-register
    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate PlotEvents block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # PlotEventsProcessor doesn't have specific validation requirements
        return []


register_processor_class(PlotEventsProcessor)








