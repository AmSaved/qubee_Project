function getCSRFToken() {
    const name = 'csrftoken';
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

let mediaRecorder = null;
let audioChunks = [];

// Update UI state
function updateUI(state, message) {
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordStatus = document.getElementById('recordStatus');
    
    if (state === 'recording') {
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        recordStatus.textContent = message || '🎤 Recording... Speak now!';
        recordStatus.style.color = '#d9534f';
    } else if (state === 'processing') {
        recordBtn.disabled = true;
        stopBtn.disabled = true;
        recordStatus.textContent = message || '🔄 Processing audio...';
        recordStatus.style.color = '#5bc0de';
    } else {
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        recordStatus.textContent = message || 'Ready to record';
        recordStatus.style.color = '#5cb85c';
    }
}

// Start recording
document.getElementById('recordBtn').addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioToServer(audioBlob, '/convert/');
            stream.getTracks().forEach(track => track.stop());
            updateUI('ready', '✅ Recording complete');
        };
        
        mediaRecorder.start();
        updateUI('recording');
        
    } catch (error) {
        console.error('Microphone error:', error);
        updateUI('ready', '❌ Microphone access denied');
        alert('Please allow microphone access to record voice.');
    }
});

// Stop recording
document.getElementById('stopBtn').addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        updateUI('processing');
    }
});

// File upload
document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioFile');
    const fileStatus = document.getElementById('fileStatus');
    
    if (fileInput.files.length === 0) {
        fileStatus.textContent = '❌ Please select a file first';
        fileStatus.style.color = '#d9534f';
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file
    if (file.size > 50 * 1024 * 1024) { // 50MB limit
        fileStatus.textContent = '❌ File too large (max 50MB)';
        fileStatus.style.color = '#d9534f';
        return;
    }
    
    fileStatus.textContent = '🔄 Processing...';
    fileStatus.style.color = '#5bc0de';
    
    await sendAudioToServer(file, '/upload/');
    
    fileStatus.textContent = `✅ ${file.name} processed`;
    fileStatus.style.color = '#5cb85c';
});

// File input change
document.getElementById('audioFile').addEventListener('change', function() {
    const fileStatus = document.getElementById('fileStatus');
    if (this.files.length > 0) {
        const file = this.files[0];
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileStatus.textContent = `📄 Selected: ${file.name} (${sizeMB} MB)`;
        fileStatus.style.color = '#5bc0de';
    } else {
        fileStatus.textContent = 'No file selected';
        fileStatus.style.color = '#6c757d';
    }
});

// Send audio to server
async function sendAudioToServer(audioData, endpoint) {
    const resultDiv = document.getElementById('result');
    const copyBtn = document.getElementById('copyBtn');
    
    // Show loading
    resultDiv.textContent = '🔄 Converting audio to Qubee text...';
    resultDiv.style.color = '#5bc0de';
    copyBtn.style.display = 'none';
    
    const formData = new FormData();
    const fieldName = endpoint === '/convert/' ? 'audio_data' : 'voice_file';
    formData.append(fieldName, audioData, 'audio.wav');
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCSRFToken(),
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            resultDiv.textContent = data.text;
            resultDiv.style.color = '#5cb85c';
            copyBtn.style.display = 'block';
        } else {
            resultDiv.textContent = `❌ Error: ${data.error}`;
            resultDiv.style.color = '#d9534f';
        }
        
    } catch (error) {
        console.error('Network error:', error);
        resultDiv.textContent = '❌ Network error. Please try again.';
        resultDiv.style.color = '#d9534f';
    }
}

// Copy text to clipboard
document.getElementById('copyBtn').addEventListener('click', async () => {
    const text = document.getElementById('result').textContent;
    const copyBtn = document.getElementById('copyBtn');
    
    try {
        await navigator.clipboard.writeText(text);
        const originalText = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        copyBtn.style.background = '#5cb85c';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
        }, 2000);
    } catch (error) {
        console.error('Copy failed:', error);
        alert('Failed to copy text to clipboard.');
    }
});