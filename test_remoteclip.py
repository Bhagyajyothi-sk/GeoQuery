import torch
import open_clip


CHECKPOINT = "models/RemoteCLIP-ViT-B-32.pt"


print("Loading OpenCLIP ViT-B/32 architecture...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=None,
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")


print("Loading RemoteCLIP checkpoint...")

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False,
)

print("\nModel loaded.")

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("\nFirst missing keys:")
    print(missing[:10])

if unexpected:
    print("\nFirst unexpected keys:")
    print(unexpected[:10])


model.eval()


# Test text encoder
text = tokenizer(
    ["satellite image of a large water body"]
)

with torch.no_grad():
    text_features = model.encode_text(text)

text_features = text_features / text_features.norm(
    dim=-1,
    keepdim=True,
)

print("\nText embedding:")
print("Shape:", text_features.shape)
print("Norm:", text_features.norm(dim=-1))


print("\nRemoteCLIP test successful.")