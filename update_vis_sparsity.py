import re
with open('experiments/cas_detr/visualize_sparsity.py', 'r') as f:
    text = f.read()

text = text.replace("n_map_to_image(feature_mask, h_feat, w_feat, padded_h, padded_w, orig_h, orig_w)",
                    "n_map_to_image(feature_mask, h_feat, w_feat, padded_h, padded_w, orig_h, orig_w, meta['scale'])")

text = text.replace("ign_map_to_image(\n                heatmap_s3, h_s3, w_s3, padded_h, padded_w, orig_h, orig_w,",
                    "ign_map_to_image(\n                heatmap_s3, h_s3, w_s3, padded_h, padded_w, orig_h, orig_w, meta['scale'],")

text = text.replace("ign_map_to_image(\n                heatmap_s4, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w,",
                    "ign_map_to_image(\n                heatmap_s4, h_s4, w_s4, padded_h, padded_w, orig_h, orig_w, meta['scale'],")

text = text.replace("ign_map_to_image(\n                heatmap_s5, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w,",
                    "ign_map_to_image(\n                heatmap_s5, h_s5, w_s5, padded_h, padded_w, orig_h, orig_w, meta['scale'],")

with open('experiments/cas_detr/visualize_sparsity.py', 'w') as f:
    f.write(text)
