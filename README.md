# product-to-pointcloud

Takes a regular product photo and turns it into a 3D point cloud. It figures out what the product is, estimates depth, removes anything hiding behind it, and spits out a colored point cloud you can view or export.

Built mainly for clean studio-style product shots — the kind with a white or light gray background.

---

## How it works

**1. Segment the product**
Converts the image to HSV and identifies background pixels as anything that's low saturation and high brightness (i.e. white/gray). Rather than blindly removing all those pixels, it flood-fills inward from the image border — so only background that's actually connected to the edges gets removed. This means white labels or gray packaging on the product itself won't get accidentally wiped.

**2. Estimate depth**
A depth map is generated for the image, giving each pixel an estimated distance from the camera.

**3. Cull occluded points**
Some depth points sit behind the product surface and wouldn't actually be visible to the camera — they're just noise from the depth estimator. These get removed by finding the nearest product surface in a local window and discarding anything significantly deeper than that.

**4. Lift to 3D**
The surviving pixels are projected from 2D image space into 3D using the depth values, producing a colored point cloud.

---

## Requirements

```
numpy
opencv-python
```

Install with:

```bash
pip install numpy opencv-python
```

---

## Usage

```python
python main.py --image product.jpg
```

Output is saved as a `.ply` point cloud file which you can open in tools like MeshLab, CloudCompare, or Three.js.

---

## Parameters

| Parameter | What it does |
|---|---|
| `SAT_THRESH` | Max saturation for a pixel to be considered background |
| `VAL_THRESH` | Min brightness for a pixel to be considered background |
| `BG_DILATE_PX` | How many pixels to grow the background mask (removes edge halos) |
| `OCCLUSION_WIN_PX` | Size of the local window when checking for occluded points |
| `OCCLUSION_TOLERANCE` | Depth tolerance before a point is considered occluded |

---

## Limitations

- Works best with plain white or light gray backgrounds. Busy or dark backgrounds will confuse the segmentation.
- Depth estimation is monocular so the scale isn't metric — it's relative.
- Very thin or transparent objects can be tricky for both segmentation and depth.

---

## Debug outputs

The pipeline saves a couple of intermediate images to help you tune the parameters:

- `debug_product_mask.png` — shows what was kept as product vs removed as background
- `debug_occlusion.png` — shows which depth points were culled as occluded

---

## License

MIT
