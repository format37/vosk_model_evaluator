import os
import pandas as pd
import jiwer
import asyncio
import websockets
import json


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
        wf = open(file_path, "rb")
        phrases = []
        while True:				
            data = wf.read(8000)
            if len(data) == 0:
                break
            await websocket.send(data)
            accept = json.loads(await websocket.recv())					
            if len(accept)>1 and accept['text'] != '':
                #accept_start = str(accept['result'][0]['start'])
                #accept_end = accept['result'][-1:][0]['end']
                accept_text = str(accept['text'])					
                phrases.append(accept_text)						
                #phrases_count += 1

        await websocket.send('{"eof" : 1}')
        accept = json.loads(await websocket.recv())
        # TODO: merge this cloned section:
        if len(accept)>1 and accept['text'] != '':
            #accept_start = str(accept['result'][0]['start'])
            #accept_end = accept['result'][-1:][0]['end']
            accept_text = str(accept['text'])            			
            phrases.append(accept_text)						
            #phrases_count += 1

    return phrases


def main():
    path = 'audio/wer/'
    files = get_files(path)
    evals_wer = []
    evals_mer = []
    evals_wil = [] 


    print('files:')
    for filename in files:
        print(path + filename)
        phrases = asyncio.get_event_loop().run_until_complete(transcribe_vosk(path + filename))
        for phrase in phrases:
            print(phrase)


if __name__ == "__main__":
	main()
