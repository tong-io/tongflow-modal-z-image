# tongflow-modal-z-image

Official TongFlow plugin. Text-to-image generation with **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Image generation** (`image-gen`) — generate an image from a text prompt.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. The Z-Image weights are public — no Hugging Face token required.
