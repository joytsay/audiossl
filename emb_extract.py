from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding
import torch

model = load_model("./models/atstframe_base.ckpt")

audio = torch.randn(1,20000) # Input audio can be of shape [1,N] or [B,1,N]

"""
extract scene (clip-level) embedding from an audio clip
=======================================
args:
    audio: torch.tensor in the shape of [1,N] or [B,1,N] 
    model: the pretrained encoder returned by load_model 
return:
    emb: retured embedding in the shape of [1,N_BLOCKS*emb_size] or [B,N_BLOCKS*emb_size], where emb_size is 768 for base model and 384 for small model.

"""
emb_scene = get_scene_embedding(audio,model)

"""
Extract frame-level embeddings from an audio clip 
==================================================
args:
    audio: torch.tensor in the shape of [1,N] or [B,1,N] 
    model: the pretrained encoder returned by load_model 
return:
    emb: retured embedding in the shape of [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS,emb_size], where emb_size is 768 for base model and 384 for small model, and T is number of (40ms) frames.
    timestamps: timestamps in miliseconds
"""
emb_timestamp,t = get_timestamp_embedding(audio,model)
print(t)


"""
By default, embeddings of 12 blocks are concatenated.

You can change N_BLOCKS 

from audiossl.methods.atstframe.embedding
embdding.N_BLOCKS=1

"""
