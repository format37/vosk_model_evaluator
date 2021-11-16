import matplotlib.pyplot as plt
import requests
import pandas as pd
import os

def send_photo_from_local_file_to_telegram(photo_path):
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT', '')
    if token == '' or chat_id == '':
        print('token or chat_id is not set. Type:')
        print('export TELEGRAM_BOT_TOKEN=your_token')
        print('export TELEGRAM_CHAT=your_chat_id')
        return None
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

evaluation_file = '~/projects/wer/evaluation.csv'
evaluation = pd.read_csv(evaluation_file, parse_dates = False)
evaluation = pd.DataFrame(evaluation[:-10])
send_report(evaluation.drop(['med_wil', 'med_wer', 'med_mer'], axis = 1, inplace = False), 'average')
send_report(evaluation.drop(['avg_wil', 'avg_wer', 'avg_mer'], axis = 1, inplace = False), 'median')