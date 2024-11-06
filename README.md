# Real-Time Captioning System

This project implements a near real-time captioning system using any of OpenAI's Whisper checkpoints and a graphical user interface (GUI) built with tkinter. The application captures audio from the microphone, transcribes it in real-time, and displays the text in a scrolling text box. The transcription process leverages OpenAI's Whisper model, enabling accurate speech-to-text transcription.

## Features
1. Real-time transcription: Captures audio continuously and displays live transcription.
2. GUI with tkinter: User-friendly interface to start, stop, and manage transcription.
3. Toggle Editing: Allows editing of transcriptions directly within the app.
4. Save Transcript: Automatically saves the transcription to a .txt file when the session ends.
5. Configurable Settings: Adjustable energy threshold, record timeout, and microphone selection.


## Requirements
- Python 3.8 or above
- tkinter (comes pre-installed with Python)
- speech_recognition
- torch
- transformers
- numpy

## Installation
- Clone this repository.
- Install the required dependencies:

    pip install torch transformers numpy speechrecognition


## Usage:

### Command-line Arguments
This system can be configured using the following arguments:

- --model_id (default: "openai/whisper-small"): Model identifier for the Whisper model on Hugging Face.
- --energy_threshold (default: 1000): Minimum volume threshold for detecting speech.
- --record_timeout (default: 2 seconds): Duration of recording before processing audio.
- --phrase_timeout (default: 3 seconds): Time between phrases.
- Example usage:

    python real_time_captioning.py --model_id "openai/whisper-small" --energy_threshold 800

### Application Usage

1. Run the application:
    python real_time_captioning.py
2. Select the microphone from the dropdown menu.

3. Click Start Transcription to begin the live transcription.

4. Use the Toggle Editing button to enable or disable text editing.

5. Click Stop Transcription to end the session and save the transcript to transcription.txt.


## Code Overview

### Key Components
- update_transcription: Updates the GUI's text box with new transcription entries and appends them to a list for saving.
- transcribe: Captures audio input, processes it using the Whisper model, and updates the transcription queue with the decoded text.
- load_model: Loads the Whisper model and processor from Hugging Face.
- setup_recorder: Configures the microphone's energy threshold for noise detection.
- main: Initializes the GUI components, including buttons for transcription control and text editing.

### GUI Elements
- Microphone Selection: Dropdown menu to select the desired microphone.
- Start/Stop Transcription: Buttons to control the transcription session.
- Toggle Editing: Enables or disables editing of transcribed text in the application.

### Saving Transcriptions
When the transcription session ends, the transcription list is saved to transcription.txt in the same directory as the script.


## Future Improvements
Potential enhancements for this project include:
- Improving error handling for unsupported microphones or issues during model loading.
- Allowing customizable output file paths for the transcription file.

