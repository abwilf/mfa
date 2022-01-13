
To get this working

```bash
    cd /work/awilf/mfa
    conda activate mfa
    export KALDI_ROOT='/work/awilf/mfa/kaldi'
    export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

    # to get working on minimal librispeech example
    # mfa align /work/awilf/mfa/Librispeech librispeech-lexicon.txt english aligned_librispeech

    # to get working on minimal siq example (once siq has been created from vtt and wavs)
    mfa align /work/awilf/mfa/siq english_dict.txt english aligned_siq
```

