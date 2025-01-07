const prefix = window.blockPrefix || "ManualClassify";  // fallback

document.addEventListener('keydown', (e) => {
    // Left arrow -> Prev
    if (e.key === "ArrowLeft") {
        const prevBtn = document.getElementById(`${prefix}-prev-button`);
        if (prevBtn) prevBtn.click();
    }
    // Right arrow -> Next
    if (e.key === "ArrowRight") {
        const nextBtn = document.getElementById(`${prefix}-next-button`);
        if (nextBtn) nextBtn.click();
    }
    // Space -> Play/Pause audio
    if (e.code === "Space") {
        e.preventDefault(); // stop page scroll
        const audio = document.getElementById(`${prefix}-audio-player`);
        if (audio) {
            if (audio.paused) {
                audio.play();
            } else {
                audio.pause();
            }
        }
    }
    // Enter -> Save classification
    if (e.key === "Enter") {
        e.preventDefault(); // avoid accidental form submits
        const saveBtn = document.getElementById(`${prefix}-save-class-button`);
        if (saveBtn) saveBtn.click();
    }
});
