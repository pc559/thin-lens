# thin-lens

When light moves past massive objects in space (such as galaxies) its trajectory bends.
This means that images coming from further away, behind the massive objects,
can be lensed, and appear warped in our sky.
See https://hubblesite.org/contents/articles/gravitational-lensing for some more discussion and
some amazing pictures of this effect!

This code was thrown together for a Cambridge open day, to demonstrate
this gravitational lensing effect.

It takes the image coming from your laptop webcam and lenses it, using the thin-lens
approximation for a particular lens configuration (some pretend galaxy) to calculate the
effect. Many thanks to Tobias Baldauf for the idea, and for fun discussions.

It requires opencv-python, and hasn't been tested on many systems, so let me know if you
have any problems running it on yours!

The main file is "webcam_lensed.py", it lenses the image coming from a webcam.
The keys l,g,c all toggle different features.
"webcam_lensed.py" should take either "lens_mapping_1000_many_lenses.npy"
or "lens_mapping_480.npy" as a command line argument.

