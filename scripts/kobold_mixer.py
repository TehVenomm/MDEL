print("Starting script, plese wait...")

import torch
import shutil
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tkinter.filedialog import askdirectory, askopenfilename

# Rubbish experiment by Concedo for KoboldAI usage
# Experimenting with the ability to blend weights from 2 LLMs of the same architecture
# Both models must have the same architecture, number of parameters, layer counts and types, and use the same vocab.

#mixer output settings
blend_ratio = 0.5           #setting to 0 gives first model, and 1 gives second model
fp16 = True                 #perform operations in fp16. Saves memory, but CPU inference will not be possible.
always_output_fp16 = True   #if true, will output fp16 even if operating in fp32
max_shard_size = "3000MiB"  #set output shard size
verbose_info = False        #will show model information when loading
force_cpu = True            #only use cpu
load_sharded = False         #load both models shard by shard

#test generation settings, only for fp32
deterministic_test = True   #determines if outputs are always the same
test_prompt = "One day,"    #test prompt for generation. only for fp32. set to empty string to skip generating.
test_max_length = 32        #test generation length


blend_ratio_b = 1.0 - blend_ratio

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

def merge_models(model1,model2):
    with torch.no_grad(): 
        tensornum = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()): 
           p1 *= blend_ratio
           p2 *= blend_ratio_b
           p1 += p2
           #print(p1)
           #print(p2)
           tensornum += 1
           if verbose_info:
               print("Merging tensor "+str(tensornum))
           pass

def read_index_filenames(sourcedir):
    index = json.load(open(sourcedir + '/pytorch_model.bin.index.json','rt'))
    fl = []  
    for k,v in index['weight_map'].items():       
        if v not in fl: 
            fl.append(v) 
    return fl
            
# def load_partial_model(sourcedir,targetfile):
#     idxfile = sourcedir + '/pytorch_model.bin.index.json'
#     shutil.copyfile(idxfile, idxfile+'.bak')
#     with open(idxfile,'rt') as infile:
#         index = json.load(infile)
#         load_dict = {}    
#         load_dict['metadata'] = index['metadata']
#         load_dict['weight_map'] = {}  
#         for k,v in index['weight_map'].items():
#             if v==targetfile:
#                 load_dict['weight_map'][k] = v
    
#     #overwrite original file
#     with open(idxfile, 'w') as outfile:
#         json.dump(load_dict, outfile)

#     #now load with the smaller file
#     mdl = AutoModel.from_pretrained(sourcedir,ignore_mismatched_sizes=True)

#     #restore original file
#     shutil.copyfile(idxfile+'.bak', idxfile)

#     #return small model
#     return mdl
    

print("Opening file dialog, please select FIRST model directory...")
model_path1 = askdirectory(title="Select Directory of FIRST model to merge")
print("Opening file dialog, please select SECOND model directory...")
model_path2 = askdirectory(title="Select Directory of SECOND model to merge")
print("Opening file dialog, please select OUTPUT model directory...")
model_path3 = askdirectory(title="Select Output Directory of merged model")
if not model_path1 or not model_path2:
    print("\nYou must select two directories containing models to merge and one output directory. Exiting.")
    exit()

with torch.no_grad(): 
    if fp16:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")
    print(device)

    # shards = read_index_filenames(model_path1)
    # model1 = load_partial_model(model_path1,shards[0])   
    # print(get_model_info(model1))
    # exit()    

    print("Loading Model 1...")
    model1 = AutoModelForCausalLM.from_pretrained(model_path1) #,torch_dtype=torch.float16 
    model1 = model1.to(device)
    model1.eval()    
    print("Model 1 Loaded. Dtype: " + str(model1.dtype))
    print("Loading Model 2...")
    model2 = AutoModelForCausalLM.from_pretrained(model_path2) #,torch_dtype=torch.float16 
    model2 = model2.to(device)
    model2.eval()    
    print("Model 2 Loaded. Dtype: " + str(model2.dtype))

    #ensure both models have the exact same layout
    m1_info = get_model_info(model1)
    m2_info = get_model_info(model2)    

    if m1_info != m2_info:
        print("Model 1 Info: " + m1_info)
        print("Model 2 Info: " + m2_info)
        print("\nERROR:\nThe two selected models are not compatible! They must have identical structure!")
        exit()
    else:
        if verbose_info:
            print("Model Info: " + m1_info)

    print("Loading a Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path1, use_fast=False)
    # Tokenize the prompt
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")

    if not fp16 and test_prompt!="":
        print("Generating for original model 1...")
        outputs = model1.generate(inputs, max_length=test_max_length, temperature=0.5, do_sample = (not deterministic_test))
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result) # Print the generated text
        print("Generating for original model 2...")
        outputs = model2.generate(inputs, max_length=test_max_length, temperature=0.5, do_sample = (not deterministic_test))
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result) # Print the generated text
    else:
        print("Skipped sample generation as running in fp16 mode")

    print("Merging models...")
    merge_models(model1,model2)
   
    if not fp16 and test_prompt!="":
        print("\nGenerating for new merged model...")
        outputs = model1.generate(inputs, max_length=test_max_length, temperature=0.5, do_sample = (not deterministic_test))
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result) # Print the generated text

    if model_path3:
        print("Saving new model...")
        newsavedpath = model_path3+"/converted_model"
        if always_output_fp16 and not fp16:
            model1.half()
        model1.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
        print("\nSaved to: " + newsavedpath)
    else:
        print("\nOutput model was not saved as no output path was selected.")

    print("\nScript Completed.")