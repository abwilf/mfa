import string
import librosa
import soundfile as sf
from textgrid import textgrid
from alex_utils import *
import re

common_sr = 16000 # need to downsample for kaldi to work

def _process_text(text_in):
    '''
    If you want to do anyting to the text that removes unnecessary elements, like "[Voiceover]" or "-", pass this function 
    '''
    text_in = text_in.replace('&gt;&gt;', '').replace('-', '')
    text_in = re.sub("[\(\[\*].*?[\)\]\*]", "", text_in) # remove all text inside [] and () and **
    text_in = re.sub('\s+',' ', text_in).strip()
    return text_in

def interval_to_seconds(interval):
    hr, minute, second = interval.split(':')
    second, microsecond = second.split('.')
    return np.round(float(int(hr)*3600 + int(minute)*60 + int(second)) + float('.'+microsecond), decimals=4)

def read_vtt(vtt_path, process_text=lambda elt: elt):
    '''
    vtt_path looks like this (and is for a whole video):
    
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
    intervals = []    
    arr = [elt for elt in open(vtt_path).read().split('Language: en')[1].split('\n') if elt != '']
    
    interval, text = None, None
    for elt in arr:
        if '-->' in elt:
            if interval is not None:
                intervals.append([interval, text])
            interval = [interval_to_seconds(subelt.strip().split(' ')[0]) for subelt in elt.split('-->')]
            text = ''
        else:
            text += ' ' + elt

    if interval is not None: # add last interval
        intervals.append([interval, text])

    for i, (_,text) in enumerate(intervals):
        intervals[i][1] = process_text(text)

    # filter intervals so left doesn't equal right (would lead to a zero wav)
    intervals = [elt for elt in intervals if elt[0][0] != elt[0][1]]
    return intervals

def split_wav(wav_in, intervals, out_dir, wav_prefix=''):
    '''
    in: wav file, intervals array in the form described by read_vtt, out_dir (directory to write subwavs to).  Will mkdirp this dir
    out: writes these segmented wavs to out_dir
    '''
    mkdirp(out_dir)

    arr, sr = librosa.load(wav_in, sr=common_sr)
    assert sr == common_sr

    sub_wavs = [arr[int(sr*left) : int(sr*right)] for left,right in list(zip(*intervals))[0]]

    # get rid of intervals that lead to an empty wav (e.g. intervals that start after the wav ends)
    zero_idxs = np.where(ar([elt.shape[0] for elt in sub_wavs])==0)[0]
    for idx in zero_idxs:
        sub_wavs.pop(idx)
        intervals.pop(idx)
    assert np.all([elt.shape[0] for elt in sub_wavs]), 'Should be no empty wavs, else messes up MFA'

    for i,sub_wav in enumerate(sub_wavs):
        sf.write(join(out_dir, '{}{:04d}.wav'.format(wav_prefix, i)), sub_wav, samplerate=sr)

    return intervals

def create_corpus(vtt_dir, wav_dir, wav_ids, corpus_dir):
    '''
    splits each wav into segments, writes first wav and label to corpus_dir/0/0-wav_id0-0000.lab & wav, 0001...etc for all segments, and 1/1-wav_id1-... for different wavs
    '''
    # Create corpus dir, split wavs
    rmrf(corpus_dir)
    mkdirp(corpus_dir)

    all_intervals = [] # intervals for all wavs
    for i,wav_id in enumerate(wav_ids):
        segment_dir_name = i
        segment_dir = join(corpus_dir, segment_dir_name)
        intervals = read_vtt(join(vtt_dir, '{}.vtt'.format(wav_id)), _process_text)

        wav_prefix = '{}-{}-'.format(segment_dir_name, wav_id)

        intervals = split_wav(join(wav_dir, '{}.wav'.format(wav_id)), intervals, segment_dir, wav_prefix=wav_prefix)
        all_intervals.append(intervals)

        for i,text in enumerate(lzip(*intervals)[1]):
            write_txt(join(segment_dir, '{}{:04d}.lab'.format(wav_prefix, i)), text)
    
    return all_intervals

def textgrid_to_subintervals(filepath):
    tg = textgrid.TextGrid.fromFile(filepath)
    intervals = [elt for elt in tg.tiers if elt.name == 'words'][0].intervals
    data = [((interval.minTime, interval.maxTime), interval.mark) for interval in intervals]
    return data
    
def align_corpus(corpus_dir, aligned_dir, all_intervals, wav_ids, _input, this_root='/work/awilf/mfa', kaldi_root='/work/awilf/kaldi'):
    rmrf(aligned_dir)

    # run MFA on segmented wavs
    prompt_str = '''
    cd {}
    export KALDI_ROOT='{}'
    export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

    mfa align {} english_dict.txt english {} --clean
    '''.format(this_root, kaldi_root, corpus_dir, aligned_dir)

    if _input:
        prompt_str = 'Please run the following command in a shell in which you have montreal force alignment installed (the `mfa align` command), then hit enter when completed: \n' + prompt_str
        _ = input(prompt_str)

    else:
        os.system(prompt_str)

    all_new_intervals = {}
    for wav_idx,wav_id in enumerate(wav_ids): # i is wav counter
        intervals = all_intervals[wav_idx]
        new_intervals = []
        for segment_idx, ((offset,_),_) in enumerate(intervals):
            textgrid_path = join(aligned_dir, wav_idx, '{}-{}-{:04d}.TextGrid'.format(wav_idx, wav_id, segment_idx))
            if not exists(textgrid_path):
                continue
            
            subintervals = textgrid_to_subintervals(textgrid_path)
            for ((start,end),word) in subintervals:
                if word != '':
                    new_intervals.append(((npr(start+offset),npr(end+offset)),word))
        
        all_new_intervals[wav_id] = {
            'wav_idx': wav_idx, 
            'intervals': new_intervals,
        }
    
    return all_new_intervals

def main(wav_dir, vtt_dir, corpus_dir, aligned_dir, intervals_path, _input):
    wav_ids = lmap(lambda elt: elt.split('/')[-1].split('.')[0], glob(join(wav_dir, '*')))
    vtt_ids = lmap(lambda elt: elt.split('/')[-1].split('.')[0], glob(join(vtt_dir, '*')))
    if not subsets_equal(wav_ids, vtt_ids):
        wav_ids = [elt for elt in wav_ids if elt in vtt_ids]
        vtt_ids = wav_ids
        print(f'There are some wav ids that are not the same as the vtt ids. We will consider only the ids that are the same between them, which are of length {len(wav_ids)}')

    all_intervals = create_corpus(vtt_dir, wav_dir, wav_ids, corpus_dir)
    new_intervals = align_corpus(corpus_dir, aligned_dir, all_intervals, wav_ids, _input)
    save_json(intervals_path, new_intervals)

    print('All done! Please see your output json file in', intervals_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='siq_wavs', type=str, required=True, help='Path to wav directory')
    parser.add_argument('--vtt_dir', default='siq_vtts', type=str, required=True, help='Path to vtt directory')
    parser.add_argument('--corpus_dir', default='_siq_corpus', type=str, required=False, help='Path to where the corpus will be created (internal)')
    parser.add_argument('--aligned_dir', default='_siq_aligned', type=str, required=False, help='Path to where the aligned corpus will be created (internal proces to mfa)')
    parser.add_argument('--intervals_path', default='intervals.json', type=str, required=True, help='Where the intervals should be written to (json format)')
    parser.add_argument('--_input', default=1, type=int, required=False, help='Use input vs os.system; do this if you need to run mfa call separately')
    args = parser.parse_args()

    main(args.wav_dir, args.vtt_dir, args.corpus_dir, args.aligned_dir, args.intervals_path, args._input)

    