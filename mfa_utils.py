import string
import librosa
import soundfile as sf
import numpy as np
import pathlib,os

def _process_text(text_in):
    '''
    If you want to do anyting to the text that removes unnecessary elements, like "[Voiceover]" or "-", pass this function 
    '''
    to_remove = [
        '[Voiceover]'
    ]
    for elt in to_remove:
        text_in = text_in.replace(elt, '')
    
    text_in = text_in.translate(str.maketrans('', '', string.punctuation))
    
    return text_in.strip()

def interval_to_seconds(interval):
    hr, minute, second = interval.split(':')
    second, microsecond = second.split('.')
    return np.round(float(int(hr)*3600 + int(minute)*60 + int(second)) + float('.'+microsecond), decimals=4)

def read_vtt(vtt_path, process_text=lambda elt: elt):
    '''
    vtt_path looks like this:
    
    WEBVTT
    Kind: captions
    Language: en

    0:00:00.000 --> 0:00:01.974
    - [Voiceover] Oh, why would
    he want to wreck something?

    0:00:01.974 --> 0:00:03.494
    That wouldn't be very nice.
    ...

    process_text function takes in a string and does what it likes with it (e.g. remove punctuation, remove [Voiceover]...etc)

    returns an array of form (interval, text), where interval is (start, end) and start,end are in their second representations (num secs.decimal)
    '''
    data = []    
    arr = [elt for elt in open(vtt_path).read().split('Language: en')[1].split('\n') if elt != '']

    interval, text = None, None
    for elt in arr:
        if '-->' in elt:
            if interval is not None:
                data.append([interval, text])
            interval = [interval_to_seconds(subelt) for subelt in elt.replace(' ','').split('-->')]
            text = ''
        else:
            text += ' ' + elt

    for i, (_,text) in enumerate(data):
        data[i][1] = process_text(text)

    return data

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def split_wav(wav_in, intervals, out_dir):
    '''
    in: wav file, intervals array in the form described by read_vtt, out_dir (directory to write subwavs to).  Will mkdirp this dir
    out: writes these wavs to out_dir
    '''
    mkdirp(out_dir)

    arr, sr = librosa.load(wav_in)
    intervals = list(zip(*intervals))[0]
    # wav_timings = [(int(sr*left), int(sr*right)) for left,right in intervals]
    sub_wavs = [arr[int(sr*left) : int(sr*right)] for left,right in intervals]

    for i,sub_wav in enumerate(sub_wavs):
        sf.write(os.path.join(out_dir, '{:04d}.wav'.format(i)), sub_wav, samplerate=sr)
    

vtt_path = '/work/awilf/social_iq_raw/transcript/waE2GdoBW68-trimmed.en.vtt'
intervals = read_vtt(vtt_path, _process_text)

wav_in = 'siq/1/temp.wav'
split_wav(wav_in, intervals, 'siq/2')
