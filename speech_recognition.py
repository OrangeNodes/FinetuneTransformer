from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
outp = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(outp) 

