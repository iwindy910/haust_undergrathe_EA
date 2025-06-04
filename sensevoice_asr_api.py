from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

def sensevoice_asr_api(audio_dir):
    model_dir = "iic/SenseVoiceSmall"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
    )


    res = model.generate(
        input=audio_dir,  
        cache={},
        language="en",  
        use_itn=False,
        batch_size_s=5,  
        merge_vad=False,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    # text = text.upper()
    if text != 'NA':
        return text
    return 'NA'


if __name__ == "__main__":
    res = sensevoice_asr_api('XXX')
    print(res)