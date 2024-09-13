import torch
from PIL import Image
import open_clip
import glob
import rich.progress

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

embeddings = []
image_files = list(glob.glob("/data/instagram/2/**/*.jpg", recursive=True))
image_files = image_files[:500]
with rich.progress.Progress() as progress:
    task = progress.add_task("[red]Embedding images...", total=len(image_files))
    for f in image_files:
        image = preprocess(Image.open(f)).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features)
        progress.update(task, advance=1)

# DBSCAN
from sklearn.cluster import HDBSCAN

embeddings = torch.cat(embeddings).cpu().numpy()
#clustering = DBSCAN(eps=0.05, min_samples=2, metric="cosine").fit(embeddings)
clustering = HDBSCAN(min_cluster_size=2, metric="euclidean").fit(embeddings)
print(clustering.labels_)
print(clustering.labels_.max())

# NEXT: find an image in the "center" of the cluster, turn it into a text description

# tsne
import numpy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=70).fit_transform(embeddings)
#plt.scatter(tsne[:, 0], tsne[:, 1])
# plot actual images
fig, ax = plt.subplots()
for i, txt in enumerate(image_files):
    im = Image.open(txt)
    # tint image by cluster
    im = im.convert("L")
    im = im.convert("RGB")
    im = numpy.array(im)
    im[:, :, 0] = im[:, :, 0] * (clustering.labels_[i] % 2)
    im[:, :, 1] = im[:, :, 1] * (clustering.labels_[i] % 2)
    im[:, :, 2] = im[:, :, 2] * (clustering.labels_[i] % 2)
    im = Image.fromarray(im)
    # rescale
    #im.thumbnail((100, 100))
    ax.imshow(im, extent=(tsne[i, 0], tsne[i, 0]+0.5, tsne[i, 1], tsne[i, 1]+0.5))
    #ax.annotate(txt, (tsne[i, 0], tsne[i, 1]))
ax.set_xlim(tsne[:, 0].min(), tsne[:, 0].max())
ax.set_ylim(tsne[:, 1].min(), tsne[:, 1].max())
# remove axis
ax.axis("off")
fig.savefig("tsne.png", dpi=1200)



"""
model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

for f in glob.glob("/data/instagram/2/4/*.jpg"):
  im = Image.open(f).convert("RGB")
  im = transform(im).unsqueeze(0)

  with torch.no_grad(), torch.cuda.amp.autocast():
    generated = model.generate(im, generation_type="top_p")

  print(f, open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))

"""
