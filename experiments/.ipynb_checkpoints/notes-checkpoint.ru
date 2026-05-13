# creating a server pipe from outside 8890 to inside 8888 of the server
ssh -p 22222 -N -L 8890:127.0.0.1:8888 ernie@66.10.218.102


# brainstorming 

Filter by relevant frequencies:  Getting of rid of frequencies not part of the targeted insturment. For Banju song it seems to be 294 to 784
    - ffmpeg -i ustadnoorbakesh.wav -af "lowpass=f=?" filtered.wav
    Lowpass and Highpass are not strict cutoffs. The following commands are
    - ffmpeg -i input.wav -af "afftfilt=real='if(between(b,80,2500),re,0)':imag='if(between(b,80,2500),im,0)'" cleaned.wav

Filtering out things whose amplitude is soft but distruptive to the model:
= ffmpeg -i input.mp3 -af "agate=threshold=0.02:ratio=10" output.mp3