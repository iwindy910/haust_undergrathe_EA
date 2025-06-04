import json
import base64
from pydub import AudioSegment
import io
from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode


API_KEY = 'XXX'
SECRET_KEY = 'XXX'

CUID = 'XXX'
RATE = 16000
DEV_PID = 1737
ASR_URL = 'http://vop.baidu.com/server_api'
SCOPE = 'audio_voice_assistant_get'
TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'


class DemoError(Exception):
    pass


def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        result_str = err.read()
    result_str = result_str.decode()

    result = json.loads(result_str)
    if 'access_token' in result.keys() and 'scope' in result.keys():
        if SCOPE and (SCOPE not in result['scope'].split(' ')):
            raise DemoError('scope is not correct')
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')


def process_audio_to_pcm(audio_path):
    sound = AudioSegment.from_wav(audio_path)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_sample_width(2)
    sound = sound.set_channels(1)

    pcm_data = io.BytesIO()
    sound.export(pcm_data, format='raw')
    pcm_data.seek(0)
    return pcm_data


def baidu_ASR(audio_path):
    token = fetch_token()
    pcm_data = process_audio_to_pcm(audio_path)
    speech_data = pcm_data.read()

    length = len(speech_data)
    if length == 0:
        raise DemoError('file length read 0 bytes')

    speech = base64.b64encode(speech_data)
    speech = str(speech, 'utf-8')
    params = {
        'dev_pid': DEV_PID,
        'format': 'pcm',
        'rate': RATE,
        'token': token,
        'cuid': CUID,
        'channel': 1,
        'speech': speech,
        'len': length
    }
    post_data = json.dumps(params, sort_keys=False)
    req = Request(ASR_URL, post_data.encode('utf-8'))
    req.add_header('Content-Type', 'application/json')
    for attempt in range(1):
        try:
            f = urlopen(req)
            result_str = f.read()
        except URLError as err:
            result_str = err.read()

        try:
            result_str = str(result_str, 'utf-8')
            result_str = eval(result_str)['result'][0]
            # result_str = result_str.upper()
            return result_str
        except:
            pass
    return 'NA'

if __name__ == '__main__':
    result = baidu_ASR(r'XXX')
    print(result)
