import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import speech_recognition as sr
import pyttsx3
import requests
import openai

conversation_memory = []  

# Function to record audio
def record_audio():
    fs = 44100  # Sample rate
    myrecording = []

    def callback(indata, frames, time, status):
        myrecording.append(indata.copy())

    print("Press 'Enter' to start recording...")
    input()  
    print("Recording started. Press 'Enter' to stop.")
    
    with sd.InputStream(samplerate=fs, channels=2, dtype='int16', callback=callback):
        input() 

    myrecording_np = np.concatenate(myrecording, axis=0)
    write('output.wav', fs, myrecording_np)
    print("Recording finished and saved as 'output.wav'.")


def recognize_speech_from_wav_openai(filename, api_key, language):
    headers = {'Authorization': f'Bearer {api_key}'}

    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'audio/wav')}
        data = {
            'model': 'whisper-1',
            'language': language 
        }
        
        response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files, data=data)

        if response.status_code == 200:
            transcription_response = response.json()
            print("Transcription successful:", transcription_response)
            return str(transcription_response) 
        else:
            print(f"Error in transcription: {response.text}")
            return None

def response_model(text):
    messages = [{"role": "system", "content": "Respond in French"}] + conversation_memory + [{"role": "user", "content": text}]
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message['content'].strip()


def speak(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 10)  # Adjust speed
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def main():
    global conversation_memory
    
    while True:
        api_key = "enter_your_own_key"
        openai.api_key = api_key
        language = 'fr'
        file_path = 'output.wav'

        # Start recording in a separate thread
        thread = threading.Thread(target=record_audio)
        thread.start()
        thread.join()  # Wait for the recording to finish

        text = recognize_speech_from_wav_openai(file_path, api_key, language) 
        output_model = response_model(text) 

        
        conversation_memory.append({"role": "user", "content": text})
        conversation_memory.append({"role": "system", "content": output_model})

        
        if len(conversation_memory) > 20:
            conversation_memory = conversation_memory[-20:]

        speak(output_model)


if __name__ == "__main__":
    main()




