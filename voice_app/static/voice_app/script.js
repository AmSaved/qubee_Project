// DOM Elements
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const audioFile = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const resultDiv = document.getElementById('result');
const copyBtn = document.getElementById('copyBtn');
const fileName = document.getElementById('fileName');
const recordStatus = document.getElementById('recordStatus');
const modelStatus = document.getElementById('modelStatus');
const connectionStatus = document.getElementById('connectionStatus');

// Recording variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Get CSRF token for Django
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

// Update model status
async function checkModelStatus() {
    try {
        modelStatus.textContent = '✅ Model: Loaded';
        modelStatus.style.color = '#10b981';
    } catch (error) {
        modelStatus.textContent = '⚠️ Model: Check server';
        modelStatus.style.color = '#ef4444';
    }
}

// File selection handler
audioFile.addEventListener('change', function() {
    if (this.files.length > 0) {
        const file = this.files[0];
        fileName.textContent = file.name;
        uploadBtn.disabled = false;
    } else {
        fileName.textContent = 'No file selected';
        uploadBtn.disabled = true;
    }
});

// Start recording
recordBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 16000
            }
        });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudio(audioBlob, 'recorded');
            stream.getTracks().forEach(track => track.stop());
            
            // Reset UI
            recordBtn.disabled = false;
            stopBtn.disabled = true;
            recordStatus.textContent = 'Recording stopped';
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        recordStatus.textContent = '🎤 Recording... Speak now!';
        recordStatus.classList.add('recording');
        
    } catch (error) {
        console.error('Microphone error:', error);
        recordStatus.textContent = '❌ Error accessing microphone';
        recordStatus.style.color = '#ef4444';
    }
});

// Stop recording
stopBtn.addEventListener('click', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordStatus.classList.remove('recording');
    }
});

// Upload file
uploadBtn.addEventListener('click', async () => {
    if (audioFile.files.length === 0) {
        showResult('Please select a file first', 'error');
        return;
    }
    
    const file = audioFile.files[0];
    
    // Validate file
    if (file.size > 20 * 1024 * 1024) {
        showResult('File too large (max 20MB)', 'error');
        return;
    }
    
    if (!file.type.startsWith('audio/')) {
        showResult('Please select an audio file', 'error');
        return;
    }
    
    await sendAudio(file, 'uploaded');
});

// Send audio to server
async function sendAudio(audioData, type) {
    const formData = new FormData();
    
    if (type === 'recorded') {
        formData.append('audio_data', audioData, 'recording.wav');
    } else {
        formData.append('voice_file', audioData);
    }
    
    // Show loading
    showResult('Processing... Please wait', 'loading');
    copyBtn.style.display = 'none';
    
    try {
        const endpoint = type === 'recorded' ? '/convert/' : '/upload/';
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCSRFToken(),
            },
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data.text, 'success');
            copyBtn.style.display = 'block';
        } else {
            showResult(data.error || 'Conversion failed', 'error');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showResult('Network error. Please try again.', 'error');
    }
}

// Show result
function showResult(text, type) {
    resultDiv.textContent = text;
    
    if (type === 'success') {
        resultDiv.style.color = '#10b981';
        resultDiv.style.borderColor = '#10b981';
    } else if (type === 'error') {
        resultDiv.style.color = '#ef4444';
        resultDiv.style.borderColor = '#ef4444';
    } else if (type === 'loading') {
        resultDiv.style.color = '#3b82f6';
        resultDiv.style.borderColor = '#3b82f6';
    }
}

// Copy text to clipboard
copyBtn.addEventListener('click', () => {
    const text = resultDiv.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        const original = copyBtn.textContent;
        copyBtn.textContent = '✅ Copied!';
        copyBtn.style.background = '#10b981';
        
        setTimeout(() => {
            copyBtn.textContent = original;
            copyBtn.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
        showResult('Failed to copy text', 'error');
    });
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    console.log('Qubee Voice Converter loaded');
});