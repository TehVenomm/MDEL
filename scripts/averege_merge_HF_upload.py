import transformers
from transformers import AutoTokenizer, WEIGHTS_NAME, CONFIG_NAME
from transformers import RobertaConfig, RobertaModel, AutoModel
from transformers.file_utils import hf_bucket_url, cached_path, hf_hub_url


# Define the input models from Huggingface hub
model_name_or_path_1 = "user/repo_1_name"
model_name_or_path_2 = "user/repo_2_name"
output_model_path = "user/output_model_name"


# Load the specified models from Huggingface hub and create the averaged model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_1)
model1 = AutoModel.from_pretrained(model_name_or_path_1)
model2 = AutoModel.from_pretrained(model_name_or_path_2)

averaged_model = (
    model1.to(torch.device("cpu")).float().eval()
    + model2.to(torch.device("cpu")).float().eval()
) / 2.0


# Store the averaged model and tokenizer
averaged_model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)


# Upload the averaged model to the Huggingface MDEL  model hub
# Define the config of the averaged model
hub_model_id = output_model_path.replace("/", "_")
model_card = {
    "model_id": hub_model_id,
    "model_name": "Averaged Model",
    "tags": ["averaged"],
    "description": "An averaged model created from two input models",
    "framework": "PyTorch",
    "language": "English",
    "license": "Apache-2.0",
    "pipeline_tag": "text-generation",
    "expert_contributors": [model_name_or_path_1, model_name_or_path_2],
}

# Upload the averaged model and its model card to the Huggingface MDEL model hub
config_path = cached_path(CONFIG_NAME, hf_bucket_url(output_model_path))
model_path = cached_path(WEIGHTS_NAME, hf_bucket_url(output_model_path))
model_hub_index_url = hf_hub_url(output_model_path)

transformers.hf_api.ModelHubMixin.push_to_hub(
    model_id=hub_model_id,
    repo_url=model_hub_index_url,
    commit_message='Add averaged model',
    model=model_path,
    tokenizer=tokenizer,
    config=config_path,
    model_card=model_card,
    access_token="<your_access_token_here>"  # add your Huggingface access token here
)

print("Averaged model created and uploaded to Huggingface MDEL model hub.")
