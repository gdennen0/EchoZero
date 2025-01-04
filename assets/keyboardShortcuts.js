document.addEventListener('keydown', (e) => {
    // Left arrow -> Prev
    if (e.key === "ArrowLeft") {
    const prevBtn = document.getElementById('prev-button');
    if (prevBtn) prevBtn.click();
    }
    // Right arrow -> Next
    if (e.key === "ArrowRight") {
    const nextBtn = document.getElementById('next-button');
    if (nextBtn) nextBtn.click();
    }
    // Space -> Play/Pause audio
    if (e.code === "Space") {
    e.preventDefault(); // stop page scroll
    const audio = document.getElementById('audio-player');
    if (audio) {
    if (audio.paused) {
    audio.play();
    } else {
    audio.pause();
    }
    }
    }
    // Enter -> Save classification (but NOT quit)
    if (e.key === "Enter") {
    e.preventDefault(); // avoid accidental form submits
    const saveBtn = document.getElementById('save-class-button');
    if (saveBtn) saveBtn.click();
    }
    });
