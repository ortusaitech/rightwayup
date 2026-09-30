# Removal requests (RightWayUp training data)

RightWayUp does not redistribute training images. The release publishes per-image manifests (`manifests/`) that
identify each image by its source ID, original URL and, where the licence requires it, its author. This page explains
how to ask us to remove an image, or its attribution, from those manifests and from future training.

## Who can ask

- Photographers or other rights holders of an image listed in `manifests/`.
- People who appear in an image or video frame used for training or evaluation.
- Licensors of a dataset we used.

## How to ask

Email **support@ortusai.io**. Please include:

1. The image URL, the manifest ID (the `id` field in `manifests/training/<family>.jsonl.gz`, in the Hugging Face
   model repository), or both.
2. Your relationship to the image: author or rights holder, person depicted, or licensor.
3. What you want: removal of the row, or removal of your name from its attribution only (CC BY 4.0 §3(a)(3)).
4. A way to reach you. You don't need to give a reason.

## What we do

- **Acknowledge within 5 business days** of receiving the request.
- **Remove the row, or its attribution if that is what you ask for, from the next manifest release**, and record the
  removal (without personal details) in `REMOVALS.md`.
- **Exclude the image from all future training** of RightWayUp models.
- **Published weights are not withdrawn** unless required by law or by a licensor's valid claim. Our position is that
  the weights do not contain copies of the training images; the model outputs only a rotation angle and a confidence.

We do not host the images themselves. To have an image taken down from the site where it is published (for example,
Flickr), please contact that site as well.

This process does not change the licence of any image, dataset or of RightWayUp itself.
