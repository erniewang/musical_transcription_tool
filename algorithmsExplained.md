Essential Melodica Algorithm (https://www.justinsalamon.com/melody-extraction.html): 
  1. equal loudness filter (biased to audible stuff. but might be tweaked to prefer a range based off of the range of the insturment being transcribed)
  2. split music into short blocks of time and run some "Discrete Fourier Transforms", which returns a set of most energetic frequencies (could be used to have some less accurate fallback pitches that could very well be the correct pitch even if it is not the correct one)
  3. run all the frequencies through a "salience function", find the salience pitch that seems to have the most matching upper harmonics. (could be maybe selectively tweaked such that only a certain pitch set will be tracked?) 
  4. contour creation: document all "pseudo melodies" or continous salience pitches for further processing.
  5. melody selection: look through all the continous, run through some math models, and return the result.

Quantization Algorithms Explained: 