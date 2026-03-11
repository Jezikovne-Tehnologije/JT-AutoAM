from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Najks/AutoAM",
    filename="best.pt",
    local_dir="../.."
)

print(f"Model downloaded to: {model_path}")