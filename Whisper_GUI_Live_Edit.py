import tkinter as tk
from tkinter import scrolledtext, Button, StringVar, OptionMenu
import threading
import argparse
import os
import numpy as np
import speech_recognition as sr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datetime import datetime
from queue import Queue
from time import sleep

def update_transcription(text_widget, queue, running, transcript_list):
    while running['status']:
        if not queue.empty():
            transcription_update = queue.get()
            text_widget.configure(state='normal')
            text_widget.insert(tk.END, transcription_update + "\n")
            if text_widget['state'] == 'disabled':
                text_widget.configure(state='disabled')
            text_widget.yview(tk.END)  # Auto-scroll
            transcript_list.append(transcription_update)  # Append transcription to the list for saving later
        sleep(0.1)

def transcribe(queue, running, args, mic_index):
    source = sr.Microphone(sample_rate=16000, device_index=mic_index)
    processor, model = load_model(args.model_id)
    recorder = setup_recorder(args.energy_threshold)

    data_queue = Queue()
    stop_listening = recorder.listen_in_background(source, lambda r, audio: data_queue.put(audio.get_raw_data()), phrase_time_limit=args.record_timeout)

    try:
        while running['status']:
            now = datetime.now()
            if not data_queue.empty():
                audio_data = b''.join(list(data_queue.queue))
                data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                inputs = processor(audio_np, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                text = processor.decode(outputs[0], skip_special_tokens=True)
                
                transcription_text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}"
                queue.put(transcription_text)

    finally:
        stop_listening()

def load_model(model_id):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    return processor, model

def setup_recorder(energy_threshold):
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = True
    return recorder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="openai/whisper-small", help="HuggingFace model identifier")
    parser.add_argument("--energy_threshold", default=1000, type=int)
    parser.add_argument("--record_timeout", default=2, type=float)
    parser.add_argument("--phrase_timeout", default=3, type=float)
    return parser.parse_args()

def main():
    args = parse_args()
    root = tk.Tk()
    root.title("Live Transcription Editor")
    
    mic_names = sr.Microphone.list_microphone_names()
    mic_var = StringVar(root)
    mic_var.set(mic_names[0])  # default value

    mic_menu = OptionMenu(root, mic_var, *mic_names)
    mic_menu.pack(padx=10, pady=10)

    text_widget = scrolledtext.ScrolledText(root, height=48, width=160)
    text_widget.pack(padx=10, pady=10)
    text_widget.configure(state='disabled')  # Start with disabled editing

    running = {'status': True}
    transcription_queue = Queue()
    transcript_list = []  # List to hold all transcripts for saving

    threads = []

    def start_transcription():
        mic_index = mic_names.index(mic_var.get())
        transcription_thread = threading.Thread(target=transcribe, args=(transcription_queue, running, args, mic_index))
        threads.append(transcription_thread)
        transcription_thread.start()
        ui_update_thread = threading.Thread(target=update_transcription, args=(text_widget, transcription_queue, running, transcript_list))
        threads.append(ui_update_thread)
        ui_update_thread.start()

    start_button = Button(root, text="Start Transcription", command=start_transcription)
    start_button.pack()

    def stop_transcription():
        running['status'] = False
        for t in threads:
            t.join()
        save_transcript(transcript_list)
        root.destroy()

    stop_button = Button(root, text="Stop Transcription", command=stop_transcription)
    stop_button.pack()

    def toggle_editing():
        if text_widget['state'] == 'normal':
            text_widget.configure(state='disabled')
        else:
            text_widget.configure(state='normal')

    edit_button = Button(root, text="Toggle Editing", command=toggle_editing)
    edit_button.pack()

    root.mainloop()

def save_transcript(transcript_list):
    with open("transcription.txt", "w") as file:
        for item in transcript_list:
            file.write(item + "\n")

if __name__ == "__main__":
    main()
