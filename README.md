# Simi-SFX: A similarity-based conditioning method for controllable sound effects synthesis

Our model is based on DDSP, conditioned on similarity scores to achieve timbre control. 
The similarity score of each sound is computed using CLAP, and is normalized within range [0,1].
By changing the value from 0 (closest to a category) to 1 (farthest to a category),
our model could synthesize variable timbres. 

