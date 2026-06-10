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
