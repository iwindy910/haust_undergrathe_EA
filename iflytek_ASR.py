import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
from pydub import AudioSegment
import io
import re

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, AudioFileBytes):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFileBytes = AudioFileBytes
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "en_us", "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url


def on_message(ws, message):
    try:
        code = json.loads(message)["code"]
        if code != 0:
            pass
        else:
            global result
            data = json.loads(message)["data"]["result"]["ws"]
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
    except Exception as e:
        print("receive msg,but parse exception:", e)


def on_error(ws, error):
    return

def on_close(ws, a, b):
    return

def on_open(ws, wsParam):
    def run(*args):
        frameSize = 8000
        intervel = 0.04
        status = STATUS_FIRST_FRAME

        with io.BytesIO(wsParam.AudioFileBytes) as fp:  # 使用 BytesIO 读取音频数据
            while True:
                buf = fp.read(frameSize)
                if not buf:
                    status = STATUS_LAST_FRAME
                if status == STATUS_FIRST_FRAME:
                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())


def audio_process(file):
    sound = AudioSegment.from_wav(file)
    sound = sound.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    byte_io = io.BytesIO()
    sound.export(byte_io, format="raw")
    byte_io.seek(0)
    return byte_io.getvalue()


def iflytek_ASR(audio_file):
    global result
    result = ""
    audio_data = audio_process(audio_file)
    wsParam = Ws_Param(APPID='XXX', APISecret='XXX',
                       APIKey='XXX',
                       AudioFileBytes=audio_data)
    for attempt in range(1):
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()
        ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.on_open = lambda ws: on_open(ws, wsParam)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        result = re.sub(r'[^\w\s]', '', result)
        # result = result.upper()
        result = result[1::]
        if result != '':
            return result
    return 'NA'


if __name__ == "__main__":
    res = iflytek_ASR('XXX')
    print(res)
