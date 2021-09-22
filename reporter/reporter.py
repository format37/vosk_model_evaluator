import os
import pandas as pd
import jiwer
import asyncio
import websockets
import json
from google.cloud import speech_v1p1beta1
from google.oauth2 import service_account


def get_files(path):
    for root, dirs, files in os.walk(path):
        files.sort()
        return files

def string_have_numbers(example):
    numbers = [str(i) for i in range(0,10)]
    for i in example:
        if i in numbers:
            return True
    return False

def evaluate(df, name):    
    wer = []
    mer = []
    wil = []
    for row in range(len(df)):
        measures = error(df.iloc[row].reference_text, df.iloc[row][name + '_text'])
        wer.append(measures['wer'])
        mer.append(measures['mer'])
        wil.append(measures['wil'])
    df[name + '_wer'] = wer
    df[name + '_mer'] = mer
    df[name + '_wil'] = wil

    return df

def error(ground_truth, hypothesis):    
    wer = jiwer.wer(ground_truth, hypothesis)
    mer = jiwer.mer(ground_truth, hypothesis)
    wil = jiwer.wil(ground_truth, hypothesis)

    # faster, because `compute_measures` only needs to perform the heavy lifting once:
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    return measures


async def transcribe_vosk(file_path):
    async with websockets.connect(os.environ.get('VOSK_SERVER_DEFAULT', '')) as websocket:
        
        phrases = []
        with open(file_path, "rb") as audio_file:
            while True:				
                data = audio_file.read(8000)
                if len(data) == 0:
                    break
                await websocket.send(data)
                accept = json.loads(await websocket.recv())					
                if len(accept)>1 and accept['text'] != '':
                    accept_text = str(accept['text'])					
                    phrases.append(accept_text)

            await websocket.send('{"eof" : 1}')
            accept = json.loads(await websocket.recv())
            if len(accept)>1 and accept['text'] != '':
                accept_text = str(accept['text'])            			
                phrases.append(accept_text)        

    return phrases


def transcribe_google(file_path):
    
    # https://cloud.google.com/speech-to-text
    # https://cloud.google.com/speech-to-text/docs/reference/rest/v1/RecognitionConfig?hl=ru#AudioEncoding
    # https://cloud.google.com/speech-to-text/docs/samples/speech-transcribe-async#speech_transcribe_async-python
    
    creds = service_account.Credentials.from_service_account_file(
        os.environ.get('GOOGLE_CREDENTIALS_FILE_PATH', ''),
        scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
    language = 'ru-RU'
    client = speech_v1p1beta1.SpeechClient(credentials=creds)
    sample_rate_hertz = 8000
    encoding = speech_v1p1beta1.RecognitionConfig.AudioEncoding.MP3

    config = {
        "language_code": language,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    #with io.open(file_path, 'rb') as audio_file:
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
        audio = speech_v1p1beta1.RecognitionAudio(content=content)

        response = client.recognize(config = config, audio = audio)
        results = []
        for result in response.results:
            alternative = result.alternatives[0]
            results.append(alternative.transcript)
    return ''.join([str(item) for item in results]).lower()


def main():
    path = 'audio/wer/'
    files = get_files(path)
    evals_wer = []
    evals_mer = []
    evals_wil = [] 


    print('vosk files:')
    for filename in files:
        print(path + filename)
        phrases = asyncio.get_event_loop().run_until_complete(transcribe_vosk(path + filename))
        text_vosk = ' '.join(phrases).replace('  ',' ')
        print(text_vosk)

    print('google files:')
    for filename in files:
        print(path + filename)        
        text_google = transcribe_google(path + filename).replace('  ',' ')
        print(text_vosk)
        


if __name__ == "__main__":
	main()
