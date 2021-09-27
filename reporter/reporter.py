import os
import pandas as pd
import jiwer
import asyncio
import websockets
import json
from google.cloud import speech_v1p1beta1
from google.oauth2 import service_account
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import time


def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.wav' in file:
                files.append(file)
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


def send_photo_from_local_file_to_telegram(photo_path):

	token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
	chat_id = os.environ.get('TELEGRAM_CHAT', '')
	session = requests.Session()
	get_request = 'https://api.telegram.org/bot' + token
	get_request += '/sendPhoto?chat_id=' + chat_id
	files = {'photo': open(photo_path, 'rb')}
	session.post(get_request, files=files)


def send_report(evaluation, description):

    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:gray']
    fig, ax = plt.subplots(1,1,figsize=(16, 9), dpi= 80)
    columns = evaluation.columns[1:]
    for i, column in enumerate(columns):
        plt.plot(evaluation.date.values, evaluation[column].values, lw=1.5, color=mycolors[i], label=column)
    plt.xticks(evaluation.date.values, rotation=60)
    plt.title(description+' error rate\nlower - better')
    plt.legend()
    plt.savefig('evaluation.png')    
    
    send_photo_from_local_file_to_telegram('evaluation.png')


def sleep_until_time(hour, minute):

    now = datetime.datetime.now()
    if now.hour > hour or (now.hour == hour and now.minute >= minute):
        tomorrow = now + datetime.timedelta(days=1)
        tomorrow = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
    else:
        tomorrow = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    time.sleep((tomorrow - now).seconds)


def main():

    # init
    path = 'audio/wer/'

    while True:

        sleep_until_time(
            int(os.environ.get('START_HOUR', '')), 
            int(os.environ.get('START_MINUTE', ''))
            )

        files = get_files(path)
        evals_wer = []
        evals_mer = []
        evals_wil = []
        #current_date = (datetime.datetime.now() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        # collect transcribations
        for filename in files:
            print('filename', filename)
            
            phrases = asyncio.get_event_loop().run_until_complete(transcribe_vosk(path + filename))
            text_vosk = ' '.join(phrases).replace('  ',' ')
            print('text_vosk', text_vosk)

            if len(text_vosk)<10:
                print('BREAK vosk len', len(text_vosk))
                os.unlink(path + filename)
                continue

            text_google = transcribe_google(path + filename).replace('  ',' ')
            print('text_google', text_google)

            if len(text_google)<10:
                print('BREAK google len', len(text_google))
                os.unlink(path + filename)
                continue
            
            if string_have_numbers(text_google):
                print('BREAK google numbers')
                os.unlink(path + filename)
                continue

            measures = error(text_google, text_vosk)
            evals_wer.append(measures['wer'])
            evals_mer.append(measures['mer'])
            evals_wil.append(measures['wil'])
            os.unlink(path + filename)
        
        # error reate evaluation
        if len(evals_wer) + len(evals_mer) + len(evals_wil) > 0:

            print('avg: wer', np.average(evals_wer), 'mer', np.average(evals_mer), 'wil', np.average(evals_wil))
            print('med: wer', np.median(evals_wer), 'mer', np.median(evals_mer), 'wil', np.median(evals_wil))

            """current = pd.DataFrame(columns = ['date', 'avg_wil', 'avg_wer', 'avg_mer', 'med_wil', 'med_wer', 'med_mer'])
            current['date'] = pd.to_datetime(current_date).date()
            current['avg_wil'] = [np.average(evals_wil)]
            current['avg_wer'] = [np.average(evals_wer)]
            current['avg_mer'] = [np.average(evals_mer)]
            current['med_wil'] = [np.median(evals_wil)]
            current['med_wer'] = [np.median(evals_wer)]
            current['med_mer'] = [np.median(evals_mer)]"""

            row = dict()
            #row['date'] = pd.to_datetime(current_date).date()
            row['date'] = current_date
            row['avg_wil'] = np.average(evals_wil)
            row['avg_wer'] = np.average(evals_wer)
            row['avg_mer'] = np.average(evals_mer)
            row['med_wil'] = np.median(evals_wil)
            row['med_wer'] = np.median(evals_wer)
            row['med_mer'] = np.median(evals_mer)
            current  = pd.DataFrame([row], columns=row.keys())

            # save
            evaluation_file = 'audio/wer/evaluation.csv'
            if os.path.isfile(evaluation_file):
                # debug
                #current.to_csv('audio/wer/debug_current_0.csv', index = False)
                evaluation = pd.read_csv(evaluation_file, parse_dates = False)                
                print('evaluation', len(evaluation))
                print('current', len(current))
                evaluation = pd.concat([evaluation, current], axis = 0)
                print('evaluation', len(evaluation))
            else:
                current.to_csv('audio/wer/debug_current_1.csv', index = False)
                print('Path does not exist', evaluation_file)
                evaluation = current
            evaluation.to_csv(evaluation_file, index = False)

            # plot and send
            start_date = pd.to_datetime((datetime.datetime.now() + datetime.timedelta(days=-10)).strftime('%Y-%m-%d'))
            evaluation.date = pd.to_datetime(evaluation.date)
            evaluation = pd.DataFrame(evaluation[evaluation.date>start_date])
            send_report(evaluation.drop(['med_wil', 'med_wer', 'med_mer'], axis = 1, inplace = False), 'average')
            send_report(evaluation.drop(['avg_wil', 'avg_wer', 'avg_mer'], axis = 1, inplace = False), 'median')

        time.sleep(60)


if __name__ == "__main__":
	main()
