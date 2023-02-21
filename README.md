# Montreal-Forced-Aligner Utils

Note: because I didn't have speaker information, I built the system without that in mind.  If you have speaker information at an utterance level, you'll want to take a different procedure but hopefully these functions are modular enough for you to use as you do that.

Note: this doesn't work if you run it in vscode b/c the os.system command falters within their environment.  Should work in terminal.

## Setup
1. Install MFA

https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

Make sure you can run their minimal examples using the `mfa align` command – that's what we'll need moving forward.

2/ Once you've done so, replace these variables in `mfa_utils.py` with your `kaldi_root` from above, and the path to this directory
```python
this_root='/work/awilf/mfa', kaldi_root='/work/awilf/kaldi'
```
Then install the dependencies we'll need on top of MFA into an environment with `python3.7`.
```bash
pip install librosa soundfile pandas numpy requests tqdm
```
These are also listed in `requirements.txt`, which you can install using `pip install -r requirements.txt`

## Minimal Example
### Minimal Example (High Level)
```bash
python mfa_utils.py --wav_dir siq_wavs --vtt_dir siq_vtts --intervals_path intervals.json
```

This code takes a set of wavs in `--wav_dir` of the form `{id}.wav` and a set of transcripts in `--vtt_dir` of the form `{id}.vtt` and performs word alignment (gets start and end timings for each word). The output is put into `--intervals_path` as a json.

### Minimal Example (Details)
To transform .vtt files (partial transcripts) and long wavs to a forced alignment (we have timestamp boundaries for the start and end of each word), we do the following steps:

1. Split the wavs into partials.  The original wavs (`siq_wavs`) and original vtt files (`siq_vtts`) are used to create the segments and corresponding segment transcripts.  These segmented wavs and transcripts go in `corpus_dir`, e.g. `siq`. This happens in `create_corpus()` in `mfa_utils.py`.
Vtt files look like this:
```
    WEBVTT
    Kind: captions
    Language: en

    0:00:00.000 --> 0:00:01.974
    - [Voiceover] Oh, why would
    he want to wreck something?

    0:00:01.974 --> 0:00:03.494
    That wouldn't be very nice.
    ...
```

In this example, we would have two segments from the original.  One corresponding to 0-1.974, the other corresponding to 1.974-3.494.

2. We align using `mfa`.  As long as your corpus is in the same format as `corpus_dir`, this code will be able to process it.  The key lines are below. The first few lines just make sure the executable `mfa` is available for your command line; the magic happens in the last line.
```bash
cd {this_root}
export KALDI_ROOT='{kaldi_root}'
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

mfa align {corpus_dir} english_dict.txt english {aligned_dir} --clean
```

3. Aligned outputs are in `aligned_dir`, e.g. `aligned_siq`. We then process and recombine the outputs into a single dictionary we save to `intervals.json`, of the form below.  This happens in `mfa_utils`::`align_corpus()`.

```json
{
    "video_id_1": {
        "wav_idx": 0, // in case you need to know where to find its outputs in siq and aligned_siq for debugging
        "intervals": [
            [
                [
                    0.333, // word start
                    0.393 // word end
                ],
                "so" // word
            ],
            ...
        ]
    },
    "video_id_2": ...
```

