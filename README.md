"pip install -e ." to export the code into the enviroment 

What is MIDI:
    - (event_time, delta_time (time passed since last event))

Data (in hexidecimal): 
    - header  : (label | size | data ) 
        -> data : | format | tracks | divisions (ticks per quarter note) |
    - track : (same as header), but data is a array of events    
    - note: delta_time, status, note, velocity

Timing: 
    - set tempo event: how many microseconds in a quarter note. 
    - ticks per quarter note, determined by BPM
Time Representation:
    - Melody Extractors can give "time data" != musical notation. 
    = Avoid quantizer drift in musical chunks

    - good libaries: librosa, essentia again, madmom, music21

Data Form:
    - the consistent form that shall be produced by these algoritmns/models will be a time series column [ time, f0, confidence, voiced? ] 
    
To write midi notes accurately within a rhythmic context in sheet music. It will need to be quantized well, and it is usually not this easy for many songs that are not straight 8ths. Take jazz for example. The rhythmn between the notes are not always *fixed intervals*, it meanders around the offbeats. 

One way to mitigate this is to run a strict quantizer, but based off of what metric. There will need to be some research done on what models can estimate the tempo of a audio recording and place downbeats. 

Users will select sections with consistent tempo and be able to establish what is the downbeat or when is it the beginning of a new measure. The tempo of a piece can change alot and can exellerate or decelerate drastically. It will not be easy pinpointing a absolute downbeat of every measure. 
