function updatePlayheadPosition(plotId, playheadId, newPosition) {
    const plot = document.getElementById(plotId);
    if (plot) {
        const playhead = plot.data.find(trace => trace.name === playheadId);
        if (playhead) {
            playhead.x0 = newPosition;
            playhead.x1 = newPosition;
            Plotly.redraw(plot);
        }
    }
}

// Update the playhead position in the Dash Store
function updateDashPlayhead(storeId, newPosition) {
    const store = document.getElementById(storeId);
    if (store) {
        store.data = newPosition;
    }
}

// Example: Move playhead for plot-1 every second
let playheadPosition1 = 0;
setInterval(() => {
    playheadPosition1 += 0.1;
    updatePlayheadPosition('plot-1', 'Playhead', playheadPosition1);
    updateDashPlayhead('plot-1-playhead-time', playheadPosition1);
}, 1000);

// Example: Move playhead for plot-2 every second
let playheadPosition2 = 0;
setInterval(() => {
    playheadPosition2 += 0.1;
    updatePlayheadPosition('plot-2', 'Playhead', playheadPosition2);
    updateDashPlayhead('plot-2-playhead-time', playheadPosition2);
}, 1000);