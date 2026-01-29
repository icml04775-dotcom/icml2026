import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(r"""
    # SAE Feature Exploration for Aerial Image Segmentation
    
    This notebook explores the learned features in a Sparse Autoencoder (SAE) integrated with a DINOv3-based segmentation model for aerial imagery.
    
    ## Key Capabilities:
    - **Feature Visualization**: Examine which features activate for different land cover classes
    - **Feature Steering**: Modify sparse activations to steer model predictions
    - **Class Profile Analysis**: Understand class-specific feature patterns
    - **Global Steering**: Propagate edits to similar regions across the image
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import os
    from pathlib import Path
    return mo, np, os, Path


@app.cell
def _(os, Path):
    # Get base directory (where this script is located)
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    return (BASE_DIR,)


@app.cell
def _(BASE_DIR):
    from model.dinov3_sae_topk_model import Dinov3TopKSAEDPTSegmentation
    import torch
    import torchvision.transforms as T
    from torchvision.transforms import v2
    import rasterio as rio
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import cv2
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import torch.nn.functional as F
    from plotly import express as px
    import altair as alt
    import pandas as pd
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Set font to Times for publication quality
    return (
        Dinov3TopKSAEDPTSegmentation,
        F,
        Image,
        Rectangle,
        T,
        alt,
        base64,
        cv2,
        go,
        make_subplots,
        pd,
        plt,
        px,
        rio,
        torch,
        v2,
    )


@app.cell
def _(os):
    os.environ['MARIMO_OUTPUT_MAX_BYTES']='20_000_000'
    return


@app.cell
def _(plt):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math
    plt.rcParams['pdf.fonttype'] = 42  # TrueType (Type-1) fonts in PDF
    plt.rcParams['ps.fonttype'] = 42   # TrueType (Type-1) fonts in PS
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration
    
    Model and inference parameters are set here. Modify these if needed.
    """)
    return


@app.cell
def _(BASE_DIR):
    # Model checkpoint path (relative to this script)
    chkpt_path = BASE_DIR / "weights" / "model_checkpoint.ckpt"
    
    # Model configuration
    sae_hidden_dim: int = 65536  # SAE dictionary size
    num_classes = 9              # Number of segmentation classes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_size = 1024              # Input image size
    patch_size = 16              # ViT patch size
    
    # Sample data path
    sample_image_path = BASE_DIR / "data" / "sample_image.tif"
    
    return (
        chkpt_path,
        device,
        img_size,
        num_classes,
        patch_size,
        sae_hidden_dim,
        sample_image_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Model
    
    Loading the trained segmentation model with SAE from checkpoint.
    """)
    return


@app.cell
def _(
    Dinov3TopKSAEDPTSegmentation,
    chkpt_path,
    device,
    img_size,
    num_classes,
    sae_hidden_dim: int,
    torch,
    v2,
):
    checkpoint = torch.load(chkpt_path, map_location=device)
    model = Dinov3TopKSAEDPTSegmentation(num_classes = num_classes, sae_hidden_dim=sae_hidden_dim).to(device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {
            k.replace('model.', ''): v 
            for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict, strict = False)
    model.eval()
    model.inference_mode()
    image_transform = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image_transform, model


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Sample Image
    """)
    return


@app.cell
def _(plt, rio, sample_image_path):
    img = None
    with rio.open(sample_image_path) as src:
        img = src.read()
    plt.imshow(img.transpose([1,2,0]))
    return (img,)


@app.cell
def _(cv2, device, image_transform, img, img_size, torch):
    img_tensor = image_transform(torch.from_numpy(img)).unsqueeze(0).to(device)
    img_rsz = cv2.resize(img.transpose([1,2,0]), (img_size, img_size), interpolation=cv2.INTER_LINEAR).transpose([2,0,1])
    img_rsz.max()
    return img_rsz, img_tensor


@app.cell
def _(model):
    backbone = model.backbone
    sae = model.sae
    head = model.segmentation_head
    return backbone, head, sae


@app.cell
def _(mo):
    mo.md(f"""
    ## Model Run
    """)
    return


@app.cell
def _(F, backbone, head, img_tensor, model, patch_size, sae, torch):
    inp = img_tensor
    with torch.no_grad():
        B, C, H, W = inp.shape

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"Image size {H}x{W} not divisible by patch size {patch_size}"
            )
        patch_h, patch_w = H // patch_size, W // patch_size
        feats = backbone(inp)
        layer_features = []
        cls_tokens = []
        for feat, cls_token in feats:
            layer_features.append(feat)
            cls_tokens.append(cls_token)
        sae_out = sae(layer_features)
        layer_reconstructed = sae_out['layer_reconstructed']

        processed_features = []
        for i, recon in enumerate(layer_reconstructed):
            processed_features.append((recon, cls_tokens[i]))

        logits = head(processed_features, patch_h, patch_w)

    class ModelSvs:
        def __init__(self, model):
            self.backbone = model.backbone
            self.sae = model.sae
            self.W_dec = self.sae.W_dec
            self.b_dec = self.sae.b_dec
            self.head = model.segmentation_head

        def get_raw_feats(self, inp):
            B, C, H, W = inp.shape
            if H % patch_size != 0 or W % patch_size != 0:
                raise ValueError(
                    f"Image size {H}x{W} not divisible by patch size {patch_size}"
                    )
            patch_h, patch_w = H // patch_size, W // patch_size
            feats = self.backbone(inp)
            layer_features = []
            cls_tokens = []
            for feat, cls_token in feats:
                layer_features.append(feat)
                cls_tokens.append(cls_token)
            return layer_features, cls_tokens

        def predict_custom_features(self, inp, w_dec, b_dec):
            z, cls_tokens = self.get_raw_feats(inp)
            h, z_concat = self.sae.encode(z, apply_topk=True)
            z_reconstructed = F.linear(h, w_dec) + b_dec
            layer_reconstructed = self.sae.split_layers(z_reconstructed)
            logits = self.get_logits(layer_reconstructed, cls_tokens)
            return logits.argmax(dim = 1)


        @torch.no_grad
        def get_sparse_coeffs(self, inp):
            feats, cls_tokens = self.get_raw_feats(inp)
            h, z_concat = self.sae.encode(feats, apply_topk=True)
            return h, cls_tokens

        @torch.no_grad
        def get_output_from_sparse_ceoff(self, h, cls_tokens):
            z_reconstructed = self.sae.decode(h)
            layer_reconstructed = self.sae.split_layers(z_reconstructed)
            logits = self.get_logits(layer_reconstructed, cls_tokens)
            return logits.argmax(dim = 1)


        def get_sae(self, z):
            h, z_concat = self.sae.encode(z, apply_topk=True)
            h_flat = h
            z_reconstructed = self.sae.decode(h)
            layer_reconstructed = self.sae.split_layers(z_reconstructed)


            return layer_reconstructed

        def get_logits(self, layer_reconstructed, cls_tokens):

            processed_features = []
            for i, recon in enumerate(layer_reconstructed):
                processed_features.append((recon, cls_tokens[i]))

            logits = self.head(processed_features, patch_h, patch_w)
            return logits

        @torch.no_grad
        def predict(self, inp):
            feats, cls_tokens = self.get_raw_feats(inp)
            layer_recon = self.get_sae(feats)
            logits = self.get_logits(layer_recon, cls_tokens)
            return logits.argmax(dim = 1)

    model_service = ModelSvs(model)
    return cls_tokens, logits, model_service, sae_out


@app.cell
def _(h_sparse, img_tensor):
    h_sparse.shape, img_tensor.max()
    return


@app.cell
def _(img_tensor, model_service, plt):
    logits_svc = model_service.predict(img_tensor)
    plt.imshow(logits_svc[0].cpu())
    return


@app.cell
def _(sae_out):
    sae_out.keys()
    return


@app.cell
def _(sae_out):
    sae_out['layer_reconstructed'][0].shape
    return


@app.cell
def _(mo):
    # Opacity slider control
    opacity_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="Mask Opacity"
    )
    return (opacity_slider,)


@app.cell
def _(mo):
    # Interactive area selection using plotly
    selection_info = mo.ui.text(
        value="No selection yet",
        label="Selected Area Info"
    )
    return


@app.cell
def _(go, img_rsz, logits, mo, np, opacity_slider):
    # Create interactive plotly visualization with CLICK-based selection

    # Prepare image data
    img_rgb = img_rsz.transpose([1, 2, 0])

    # Normalize to 0-255 range if needed
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb = img_rgb.astype(np.uint8)

    # Create mask overlay
    mask_data = logits[0].argmax(dim=0).cpu().numpy()

    # Create figure - using scatter for better click detection
    fig_interactive = go.Figure()

    # Add image as background
    fig_interactive.add_trace(
        go.Image(z=img_rgb, name="image")
    )

    # Add mask as heatmap overlay with better click support
    fig_interactive.add_trace(
        go.Heatmap(
            z=mask_data,
            colorscale='Viridis',
            opacity=opacity_slider.value,
            showscale=True,
            name="segmentation",
            hovertemplate='Class: %{z}<br>x: %{x}<br>y: %{y}<extra></extra>',
            customdata=mask_data  # Store mask data for access on click
        )
    )


    # Make it interactive with marimo - captures click events
    plotly_chart = mo.ui.plotly(fig_interactive)
    return img_rgb, plotly_chart


@app.cell
def _(mo, opacity_slider):
    mo.md(f"""
    ### Interactive Visualization Controls\n\nAdjust mask opacity: {opacity_slider}
    """)
    return


@app.cell
def _(mo, opacity_slider, plotly_chart):
    mo.md(f"""
    ### Interactive Visualization

    **Instructions:**
    - Click on the image to inspect specific points
    - Use the manual selection controls below to define a region of interest

    {opacity_slider}

    {plotly_chart}
    """)
    return


@app.cell
def _(device, img_size, plt, torch):
    x_left, y_bottom = 630, 90
    w_patch, h_patch = 120, 90
    region_mask = torch.zeros(img_size, img_size).to(device)
    region_mask[y_bottom:y_bottom+h_patch, x_left:x_left+w_patch] = 1
    plt.imshow(region_mask.cpu())
    return h_patch, region_mask, w_patch, x_left, y_bottom


@app.cell
def _(h_patch, img_rgb, plt, w_patch, x_left, y_bottom):
    img_patch = img_rgb[y_bottom:y_bottom+h_patch, x_left:x_left+w_patch]
    plt.imshow(img_patch)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Extract Features
    """)
    return


@app.cell
def _(sae_out):
    sae_out.keys()
    return


@app.cell
def _(sae_out):
    h_sparse = sae_out['h_sparse']
    n_tokens = h_sparse.shape[1]
    h_token = w_token = int((n_tokens)**0.5)
    sparse_coeffs = h_sparse.reshape(h_sparse.shape[0], h_token, w_token, -1)
    h_sparse.shape, sparse_coeffs.shape
    return h_sparse, h_token, sparse_coeffs, w_token


@app.cell
def _(F, h_token, logits, w_token):
    mask_patch_shape = F.interpolate(logits.argmax(dim=1).unsqueeze(1).float(), size = (h_token, w_token), mode = 'nearest').squeeze(1).cpu()
    mask_patch_shape.shape
    return


@app.cell
def _(F, h_token, region_mask, w_token):
    region_mask_patches = F.interpolate(
                region_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(h_token, w_token),
                mode='nearest'
            ).squeeze().bool()
    region_mask_patches.shape
    return (region_mask_patches,)


@app.cell
def _(mo):
    mo.md(rf"""
    You get the number of patchs and sparse coefficients for each patch
    """)
    return


@app.cell
def _(region_mask_patches, sparse_coeffs, torch):
    region_features = sparse_coeffs.squeeze()[region_mask_patches]
    mean_features = region_features.mean(dim=0) 
    threshold = 0.01 * mean_features.max()
    active_mask = mean_features > threshold
    active_indices = active_mask.nonzero().squeeze(-1).tolist()
    region_top_values, region_top_indices = torch.topk(mean_features, 64)
    region_features.shape, threshold, region_top_values, region_top_indices
    return region_features, region_top_indices, region_top_values


@app.cell
def _(logits):
    logits.argmax(dim=1).shape
    return


@app.cell
def _(F, device, logits, sparse_coeffs, torch):
    @torch.no_grad()
    def compute_class_feature_profiles(
        sparse_coeffs,
        seg_labels,
        num_classes: int = 9
    ) -> torch.Tensor:
        class_profiles = torch.zeros(num_classes, sparse_coeffs.shape[-1], device=device)
        class_counts = torch.zeros(num_classes, device=device)

        # for i, image in enumerate(images):

        sparse = sparse_coeffs
        B, h, w, D = sparse.shape


        # Resize labels to feature map size
        if seg_labels.shape[-2:] != (h, w):
            seg_labels = F.interpolate(
                    seg_labels.unsqueeze(1).float(),
                    size=(h, w),
                    mode='nearest'
                ).squeeze(1).long()

        seg_labels = seg_labels.squeeze(0)  # (h, w)
        sparse_flat = sparse.squeeze(0)  # (h, w, d_hidden)

        # Accumulate for each class
        for cls_idx in range(num_classes):
            mask = (seg_labels == cls_idx)
            if mask.any():
                cls_features = sparse_flat[mask]  # (N_cls, d_hidden)
                print(sparse_flat.shape, cls_features.shape)
                class_profiles[cls_idx] += cls_features.sum(dim=0)
                class_counts[cls_idx] += mask.sum()

        # Average
        class_profiles = class_profiles / (class_counts.unsqueeze(1) + 1e-8)
        print(class_counts)
        return class_profiles

    class_profiles = compute_class_feature_profiles(sparse_coeffs, logits.argmax(dim=1))
    dominant_class = class_profiles.argmax(dim=0)
    class_profiles.shape, dominant_class.shape
    return (class_profiles,)


@app.cell
def _(class_profiles):
    class_profiles.max()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sparse distribution
    The scatter plot shows that different classes are getting activated through different features
    """)
    return


@app.cell
def _(class_profiles):
    class_profiles[:,0:10000].shape
    return


@app.cell
def _(class_profiles, px):
    fig_cp = px.imshow(class_profiles[:,0:60000].cpu().numpy(), 
                    color_continuous_scale='hot',
                    aspect='auto',
                    labels={'x': 'Feature', 'y': 'Row', 'color': 'Value'},
                     zmin = 0, zmax = 3 )
    fig_cp.update_layout(width=1200, height=400)
    fig_cp.show()
    return


@app.cell
def _(region_top_indices, region_top_values):
    region_top_indices, region_top_values
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Extract class region features
    """)
    return


@app.cell
def _(class_profiles, region_mask_patches, sparse_coeffs, torch):
    def get_region_class_profile_match(class_profile, sparse_coeffs, region_mask, top_k = 32):
        region_features = sparse_coeffs.squeeze()[region_mask]
        mean_features = region_features.mean(dim=0) 
        threshold = 0.1 * mean_features.max()
        active_mask = mean_features > threshold
        region_top_indices = active_mask.nonzero().squeeze(-1)
        if isinstance(region_top_indices, int):
            region_top_indices = [region_top_indices]
        region_top_values = mean_features[active_mask]
        
        region_class_match_index = []
        region_class_match_value = []
        for idx in range(len(class_profile)):
            class_top_val, class_top_index = torch.topk(class_profile[idx], top_k)
            region_top_class_mask = torch.isin(class_top_index, region_top_indices)
            region_class_match_index.append(class_top_index[region_top_class_mask])
            region_class_match_value.append(class_top_val[region_top_class_mask])
        return region_class_match_index, region_class_match_value

    def get_discriminating_features_exclusive(class_profile, sparse_coeffs, region_mask, top_k=32):
        """
        Find class-exclusive features using set operations
        """
        region_features = sparse_coeffs.squeeze()[region_mask]
        mean_features = region_features.mean(dim=0) 
        region_top_values, region_top_indices = torch.topk(mean_features, top_k)

        region_class_match_index = []
        region_class_match_value = []

        # Get top-k for all classes
        all_class_tops = []
        for idx in range(len(class_profile)):
            _, class_top_index = torch.topk(class_profile[idx], top_k)
            all_class_tops.append(set(class_top_index.tolist()))

        # For each class
        for idx in range(len(class_profile)):
            class_top_val, class_top_index = torch.topk(class_profile[idx], top_k)

            # Region features
            region_set = set(region_top_indices.tolist())

            # This class features
            class_set = set(class_top_index.tolist())

            # All OTHER classes features (union)
            other_classes_set = set()
            for other_idx in range(len(class_profile)):
                if other_idx != idx:
                    other_classes_set.update(all_class_tops[other_idx])

            # Exclusive features = (region ∩ this_class) - other_classes
            exclusive_set = (region_set & class_set) - other_classes_set

            # Convert back to tensor and get values
            if len(exclusive_set) > 0:
                exclusive_list = list(exclusive_set)
                exclusive_indices = torch.tensor(exclusive_list, device=region_top_indices.device)

                # Get values for these indices
                exclusive_values = []
                for feat_idx in exclusive_indices:
                    # Find where this feature appears in region_top_indices
                    mask = region_top_indices == feat_idx
                    if mask.any():
                        exclusive_values.append(region_top_values[mask][0])
                    else:
                        exclusive_values.append(mean_features[feat_idx])
                exclusive_values = torch.tensor(exclusive_values, device=region_top_indices.device)
            else:
                exclusive_indices = torch.tensor([], device=region_top_indices.device, dtype=torch.long)
                exclusive_values = torch.tensor([], device=region_top_indices.device)

            region_class_match_index.append(exclusive_indices)
            region_class_match_value.append(exclusive_values)

        return region_class_match_index, region_class_match_value

    get_region_class_profile_match(class_profiles, sparse_coeffs, region_mask_patches), get_discriminating_features_exclusive(class_profiles, sparse_coeffs, region_mask_patches, 32)
    return


@app.cell
def _(px, region_mask_patches):
    px.imshow(region_mask_patches.cpu())
    return


@app.cell
def _(h_sparse):
    (h_sparse[0]>1e-6).shape
    return


@app.cell
def _(h_sparse, torch):
    torch.topk((h_sparse[0]>1e-6).sum(dim = 0), 20)
    return


@app.cell
def _(class_profiles, mo, px, sparse_coeffs, torch):
    print(torch.topk(class_profiles[4], 15))
    sparse_coeff_plot = mo.ui.plotly(px.imshow(sparse_coeffs[0][:,:,617].cpu()))
    return (sparse_coeff_plot,)


@app.cell
def _(mo, sparse_coeff_plot):
    mo.md(f"""
    In this example plot, it is clearly seen that the the sparse coeeficient that belongs to a given class is activating on all pixels covered by vegetation. Vegetation is class 5
    {sparse_coeff_plot}
    """)
    return


@app.cell
def _(F, np, plt, sparse_coeffs):
    def overlay_image_sparse_coeff(image_rgb, sparse_coeff, sparse_idxs, colormap='viridis', alpha=0.5, title="Overlay", ax=None,):
        """
        Overlays a mask on an RGB image using a specific Matplotlib colormap.
        Background (0) values in the mask are treated as transparent.

        Parameters:
        - image_rgb: numpy array of shape (H, W, 3).
        - mask: numpy array of shape (H, W). Can be binary, integer labels, or float heatmap.
        - colormap: string, Matplotlib colormap name (e.g., 'viridis', 'turbo', 'plasma').
        - alpha: float (0 to 1), global transparency of the overlay.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Show the Base Image
        ax.imshow(image_rgb)
        sparse_image = sparse_coeffs[:,:,:,sparse_idxs[0]]
        for sparse_idx in sparse_idxs[1:]:
            sparse_image = sparse_image+sparse_coeffs[:,:,:,sparse_idxs[1]]

        mask = F.interpolate(sparse_image.unsqueeze(0)/sparse_image.max(), size=(image_rgb.shape[1], image_rgb.shape[1]),mode='nearest').cpu()[0][0]
        # 2. Mask the background (where mask is 0)
        # This creates a "masked array" where 0s are invalid and won't be drawn
        masked_data = np.ma.masked_where(mask == 0, mask)

        # 3. Overlay the Mask using the chosen colormap
        # 'alpha' here controls the transparency of the colored parts only
        im = ax.imshow(masked_data, cmap=colormap, alpha=alpha, vmin=0, vmax=1.0)



        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mask Intensity', rotation=270, labelpad=15)



        ax.set_title(title)
        ax.axis('off')
        return im
    return (overlay_image_sparse_coeff,)


@app.cell
def _(F, img_rgb, sparse_coeffs):
    F.interpolate(sparse_coeffs[:,:,:,617].unsqueeze(0), size=(1024, 1024),mode='nearest').cpu()[0][0].shape, img_rgb.shape
    return


@app.cell
def _(img_rgb, overlay_image_sparse_coeff, sparse_coeffs):
    overlay_image_sparse_coeff(img_rgb, sparse_coeffs,  [59138], colormap='viridis')
    return


@app.cell
def _(img_rgb, overlay_image_sparse_coeff, plt, sparse_coeffs):
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image")
    axes[0].axis('off')
    overlay_image_sparse_coeff(img_rgb, sparse_coeffs,  [13128, 14668], colormap='viridis', ax = axes[1], title = "Tree Activations")
    overlay_image_sparse_coeff(img_rgb, sparse_coeffs,  [38331, 23403], colormap='viridis', ax = axes[2], title="Road Activations")
    overlay_image_sparse_coeff(img_rgb, sparse_coeffs,  [5368,7230], colormap='viridis', ax = axes[3], title="Building activations")
    return


@app.cell
def _(img_rgb, overlay_image_sparse_coeff, sparse_coeffs):
    overlay_image_sparse_coeff(img_rgb, sparse_coeffs,  [5368,7230], colormap='viridis')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Steer Features
    """)
    return


@app.cell
def _(cls_tokens, h_sparse, model_service):
    model_service.get_output_from_sparse_ceoff(h_sparse, cls_tokens).shape
    return


@app.cell
def _(region_mask_patches, sparse_coeffs):
    sparse_coeffs.shape, region_mask_patches.shape
    return


@app.cell
def _(class_profiles, torch):
    class_profiles[4]
    torch.topk(class_profiles[4], 5)
    return


@app.cell
def _(region_mask_patches, sparse_coeffs, torch):
    sparse_coeffs_cp = sparse_coeffs.clone()
    sparse_coeffs_flat = sparse_coeffs_cp.view(1, -1, 16384)
    feature_indices_target = [8082, 452]
    feature_indices_supress = [ 2693]
    for feat_idx in feature_indices_target:
        sparse_coeffs_flat[0, region_mask_patches.view(-1), feat_idx] += torch.tensor(5000.0)
    for feat_idx in feature_indices_supress:
        sparse_coeffs_flat[0, region_mask_patches.view(-1), feat_idx] = 0.00
    return (sparse_coeffs_flat,)


@app.cell
def _(px, sparse_coeffs_flat):
    px.imshow(sparse_coeffs_flat[0][:,8082].view(64,64).cpu())
    return


@app.cell
def _(region_mask_patches, sparse_coeffs_flat):
    sparse_coeffs_flat[0, region_mask_patches.view(-1), 14634], sparse_coeffs_flat.shape
    return


@app.cell
def _(cls_tokens, model_service, px, sparse_coeffs_flat):
    px.imshow(model_service.get_output_from_sparse_ceoff(sparse_coeffs_flat, cls_tokens).cpu()[0]) 
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Gap Filling
    """)
    return


@app.cell
def _(
    F,
    class_profiles,
    cls_tokens,
    model_service,
    px,
    region_features,
    region_mask_patches,
    sparse_coeffs,
):
    target_class_idx = 5
    target_profile = class_profiles[target_class_idx]
    region_mean = region_features.mean(dim = 0)
    gap = F.relu(target_profile - region_mean)
    sparse_coeffs_gap = sparse_coeffs.clone().view(1, -1, sparse_coeffs.shape[-1])
    sparse_coeffs_gap[0, region_mask_patches.view(-1)] += 10*gap
    px.imshow(model_service.get_output_from_sparse_ceoff(sparse_coeffs_gap, cls_tokens).cpu()[0]) 
    return region_mean, sparse_coeffs_gap, target_class_idx, target_profile


@app.cell
def _(
    cls_tokens,
    go,
    h_patch,
    img_rgb,
    img_tensor,
    make_subplots,
    model_service,
    sparse_coeffs_gap,
    w_patch,
    x_left,
    y_bottom,
):
    fig_gap_filling = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Base Image", "Model prediction", "Steered Output"),
    )

    # Left Plot (Row 1, Col 1): Just the Base Image
    fig_gap_filling.add_trace(
        go.Image(z=img_rgb),
        row=1, col=1
    )
    fig_gap_filling.add_trace(
        go.Heatmap(z=model_service.predict(img_tensor).cpu()[0].numpy(), colorscale='viridis'),
        row=1, col=2
    )
    fig_gap_filling.update_yaxes(autorange="reversed", row=1, col=2)


    # Right Plot (Row 1, Col 2): Base Image...
    fig_gap_filling.add_trace(
        go.Heatmap(z=model_service.get_output_from_sparse_ceoff(sparse_coeffs_gap, cls_tokens).cpu()[0].numpy(), colorscale='viridis'),
        row=1, col=3
    )
    fig_gap_filling.update_yaxes(autorange="reversed", row=1, col=3)

    fig_gap_filling.add_shape(
        type="rect",
        x0=x_left, y0=y_bottom+h_patch,  # Bottom-left corner (or Top-left for images)
        x1=x_left+w_patch, y1=y_bottom,  # Top-right corner (or Bottom-right for images)
        line=dict(color="Red", width=3),
        fillcolor="rgba(255, 0, 0, 0.0)", # Optional transparent fill,
        row=1, col=1
    )
    fig_gap_filling.update_layout(
        width=1200,
        autosize=True,
        title="Steering the model output towards desired class (from bare to tree class)"
    )
    fig_gap_filling
    return


@app.cell
def _(
    F,
    cls_tokens,
    mo,
    model_service,
    px,
    region_mask_patches,
    sparse_coeffs,
    target_profile,
    torch,
):
    @torch.no_grad()
    def propagate_steering_to_similar_regions(
        sparse_coeffs: torch.Tensor,      # (1, h, w, d_hidden)
        region_mask: torch.Tensor,         # (h, w) bool - your selected region
        target_class_profile: torch.Tensor, # (d_hidden,) - target class features
        similarity_threshold: float = 0.5,  # cosine similarity threshold
        top_k_features: int = 64,          # number of features to consider
    ):
        """
        Find all pixels similar to selected region and apply gap-filling.
        """
        h, w, d_hidden = sparse_coeffs.shape[1], sparse_coeffs.shape[2], sparse_coeffs.shape[3]
        sparse_flat = sparse_coeffs.squeeze(0).reshape(-1, d_hidden)  # (h*w, d_hidden)

        # Step 1: Get feature signature of selected region
        region_mask_flat = region_mask.view(-1)  # (h*w,)
        region_features = sparse_flat[region_mask_flat]  # (N_region, d_hidden)
        region_signature = region_features.mean(dim=0)   # (d_hidden,) - prototype

        # Step 2: Find which features define this region (top-k active)
        top_vals, top_indices = torch.topk(region_signature, top_k_features)

        # Step 3: Compute similarity of ALL pixels to region signature
        # Only compare on the active features (avoids zero-dilution)
        region_active = region_signature[top_indices]  # (top_k,)
        all_pixels_active = sparse_flat[:, top_indices]  # (h*w, top_k)

        # Cosine similarity between each pixel and region prototype
        similarity = F.cosine_similarity(
            all_pixels_active, 
            region_active.unsqueeze(0),  # (1, top_k)
            dim=-1
        )  # (h*w,)

        # Step 4: Create mask of similar pixels
        similar_mask = similarity > similarity_threshold  # (h*w,)

        # Step 5: Apply gap-filling ONLY to similar pixels
        gap = F.relu(target_class_profile - region_signature)  # (d_hidden,)

        # Clone and modify
        sparse_modified = sparse_flat.clone()
        sparse_modified[similar_mask] = sparse_modified[similar_mask] + 5*gap

        return sparse_modified.unsqueeze(0), similar_mask.reshape(h, w)

    sparse_coeffs_all_region, similar_mask = propagate_steering_to_similar_regions(sparse_coeffs, region_mask_patches, target_profile)

    steered_output = model_service.get_output_from_sparse_ceoff(sparse_coeffs_all_region, cls_tokens).cpu()[0]

    global_steering = px.imshow(steered_output, color_continuous_scale='viridis')
    gap_steering_fig = mo.ui.plotly(global_steering)
    return (
        gap_steering_fig,
        similar_mask,
        sparse_coeffs_all_region,
        steered_output,
    )


@app.cell
def _(gap_steering_fig, mo):
    mo.md(f"""
    The area belonging to bareland was successfully converted into forest area
    {gap_steering_fig}
    """)
    return


@app.cell
def _(F, img_rgb, np, plt, similar_mask, steered_output):
    def plot_global_steering(image, similar_mask, steered_output, alpha = 0.6):
        fig, axs = plt.subplots(1,2, figsize=(10, 5))
        axs[0].imshow(image)

        mask = F.interpolate(similar_mask.float().unsqueeze(0).unsqueeze(0), size=(1024,1024),mode='nearest').squeeze().cpu()
        # 2. Mask the background (where mask is 0)
        # This creates a "masked array" where 0s are invalid and won't be drawn
        masked_data = np.ma.masked_where(mask == 0, mask)
        print(image.shape, masked_data.shape)
        # 3. Overlay the Mask using the chosen colormap
        # 'alpha' here controls the transparency of the colored parts only
        axs[0].imshow(masked_data, cmap = 'viridis', alpha=alpha, vmin=0.9, vmax=2.0)
        axs[0].set_title("Image patches similar to selected region")
        axs[1].imshow(steered_output)
        axs[1].set_title("Steered Output")
        return fig

    steered_figure = plot_global_steering(img_rgb, similar_mask, steered_output)
    return


@app.cell
def _(sparse_coeffs_all_region):
    sparse_coeffs_all_region.shape
    return


@app.cell
def _(img_tensor, model_service, plt):
    plt.imshow(model_service.predict(img_tensor).cpu()[0])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Steer Decoder Features
    """)
    return


@app.cell
def _(class_profiles, torch):
    torch.topk(class_profiles[5], 5), torch.topk(class_profiles[2], 5)
    return


@app.cell
def _(region_features, torch):
    torch.topk((region_features > 1e-6).sum(dim=0), 32)
    return


@app.cell
def _(class_profiles):
    class_profiles[[0,1,2,3,4,6,7,8]].sum().shape, class_profiles.shape
    return


@app.cell
def _(class_profiles, region_features, target_class_idx, torch):

    target_top_val, target_top_idx = torch.topk(class_profiles[5],32)
    other_top_val, other_top_idx = [], []
    for idx in range(len(class_profiles)):
        if idx == target_class_idx:  # ← Skip the target class!
            continue
        class_top_val, class_top_idx = torch.topk(class_profiles[idx],32)
        other_top_val.append(class_top_val)
        other_top_idx.append(class_top_idx)

    other_top_val = torch.cat(other_top_val)
    other_top_idx = torch.cat(other_top_idx)
    region_top_val, region_top_idx = torch.topk(region_features.max(dim = 0)[0], 32)


    target_exclusive_set = ((set(target_top_idx.cpu().tolist()) - set(other_top_idx.cpu().tolist())))
    target_exclusive = [i for i in target_top_idx.cpu().tolist() if i in target_exclusive_set]
    return target_exclusive, target_top_idx


@app.cell
def _(target_exclusive):
    target_exclusive
    return


@app.cell
def _(target_top_idx):
    (target_top_idx)
    return


@app.cell
def _(region_mean, target_profile, torch):
    set(torch.topk(region_mean, 32)[1].cpu().tolist()) - set(torch.topk(target_profile, 32)[1].cpu().tolist())
    return


@app.cell
def _(img_tensor, model_service, plt, target_exclusive):
    w_dec = model_service.W_dec.clone()
    b_dec = model_service.b_dec.clone()
    w_dec.shape, b_dec.shape

    features_to_ablate = [17664, 33665, 59138, 24196, 35847, 51465, 27534, 40347, 55836, 22941, 9506, 16295, 36903, 37041, 52916, 47668, 7866, 26435, 11214, 49103, 13648, 13153, 56676, 15462, 9581, 9453, 62705, 54643, 7414, 31991, 23803, 61181]
    w_dec[:, features_to_ablate] = 6.0*w_dec[:,target_exclusive].mean(dim=1).unsqueeze(1)

    (w_dec[:,6260] - w_dec[:,13091]).abs().sum()
    plt.imshow(model_service.predict_custom_features(img_tensor, w_dec, b_dec).cpu()[0])
    plt.axis("off")
    return (w_dec,)


@app.cell
def _(target_exclusive, w_dec):
    w_dec[:,target_exclusive].mean(dim=1).shape
    return


@app.cell
def _(w_dec):
    w_dec[:,2429], w_dec[:,13091]
    return


if __name__ == "__main__":
    app.run()
