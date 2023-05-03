import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tkinter.filedialog import askdirectory, askopenfilename


# Convert a LLM to fp16/fp32 and shard it

fp16 = True                     #store as fp16 if true and fp32 if not
max_shard_size = "2000MiB"     #will split file into shards of this size
verbose_info = True             #will show model information when loading

def get_model_info(model):
    with torch.no_grad(): 
        outfo = ""
        cntent = 0
        outfo += "\n==============================\n"
        for name, para in model.named_parameters():
            cntent += 1
            outfo += ('{}: {}'.format(name, para.shape))+"\n"
        outfo += ("Num Entries: " + str(cntent))+"\n"
        outfo += ("==============================\n")
        return outfo


print("Opening file dialog, please select model directory...")
model_path1 = askdirectory(title="Select Directory of model")
print("Opening file dialog, please select output directory...")
model_path2 = askdirectory(title="Select Directory of output")
if not model_path1:
    print("\nYou must select directories containing model. Exiting.")
    exit()

with torch.no_grad(): 
    print("Loading Models...")
    if fp16:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    model1 = AutoModelForCausalLM.from_pretrained(model_path1)
    model1.eval()    
    print("Model 1 Loaded. Dtype: " + str(model1.dtype))

    m1_info = get_model_info(model1)     
    if verbose_info:
        print("Model Info: " + m1_info)

    if model_path2:
        print("Saving new model...")
        newsavedpath = model_path2+"/converted_model"        
        model1.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
        print("\nSaved to: " + newsavedpath)
    else:
        print("\nOutput model was not saved as no output path was selected.")

    print("\nScript Completed.")