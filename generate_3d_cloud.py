import cv2
import torch
import numpy as np
import trimesh
from depth_anything_v2.dpt import DepthAnythingV2

IMG_PATH    = r'C:\Users\shabd\Desktop\Tequila\Model\Depth-Anything-V2\img.jpeg'
CHECKPOINT  = r'C:\Users\shabd\Desktop\Tequila\Model\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth'
OUTPUT_PLY  = r'C:\Users\shabd\Desktop\Tequila\Model\Depth-Anything-V2\output_cloud.ply'
OUTPUT_BEV  = r'C:\Users\shabd\Desktop\Tequila\Model\Depth-Anything-V2\bev_map.png'

INFER_WIDTH = 1280
MAX_DEPTH_M = 5.0
FOV_H_DEG   = 70.0
VOXEL_SIZE  = 0.02
# if a pixel is very unsaturated and reasonably bright, it's probably background
# anything this gray or close to white we'll treat as background
SAT_THRESH    = 22
VAL_THRESH    = 45
BG_DILATE_PX  = 10
BG_FILL_COLOR = (114, 114, 114)

# how forgiving we are when deciding if a point is hiding behind something else
OCCLUSION_TOLERANCE = 0.35
OCCLUSION_WIN_PX    = 15

# walls tend to be far away and perfectly flat — these thresholds catch that
REMOVE_WALLS             = True
WALL_DEPTH_PERCENTILE    = 78
WALL_FLATNESS_THRESHOLD  = 0.018
WALL_LOCAL_RADIUS        = 20

# only useful if you're navigating a floor plan instead of scanning an object
FLOOR_NAV_MODE = False
CAMERA_HEIGHT  = 1.5
HEIGHT_MIN, HEIGHT_MAX = 0.0, 2.5


def voxel_downsample(pts, colors, voxel_size):
    # snap each point to the nearest grid cell and keep one representative per cell
    # this cuts down millions of redundant points without losing visible detail
    vox = np.floor(pts / voxel_size).astype(np.int64)

    # pack x, y, z grid indices into a single integer so we can deduplicate cheaply
    keys = (vox[:, 0] * 1_000_000_000 +
            vox[:, 1] * 1_000_000     +
            vox[:, 2])

    _, first = np.unique(keys, return_index=True)
    return pts[first], colors[first]


def segment_product(bgr):
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # pixels that are nearly white or gray are candidates for background
    candidate  = ((s < SAT_THRESH) & (v > VAL_THRESH)).astype(np.uint8)
    filled     = candidate.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # walk the image border and flood-fill outward — whatever connects to the edge is background
    border = ([(0, x) for x in range(w)] + [(h-1, x) for x in range(w)] +
              [(y, 0) for y in range(h)] + [(y, w-1) for y in range(h)])
    for y, x in border:
        if filled[y, x] == 1:
            cv2.floodFill(filled, flood_mask, (x, y), 2)

    bg_mask = (filled == 2).astype(np.uint8)

    # grow the background mask slightly so we don't leave a thin halo around the product
    if BG_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (BG_DILATE_PX*2+1,)*2)
        bg_mask = cv2.dilate(bg_mask, k)

    product_mask = ((1 - bg_mask) * 255).astype(np.uint8)
    print(f"  Segmentation -> {(product_mask>0).sum():,} fg  "
          f"| {(bg_mask>0).sum():,} bg removed")
    cv2.imwrite('debug_product_mask.png', product_mask)
    return product_mask


def raycast_occlusion_mask(depth_m, product_mask):
    # anything deeper than the nearest product surface (plus a small tolerance) is hiding behind it
    sentinel   = depth_m.max() + 1.0
    prod_depth = np.where(product_mask > 0, depth_m, sentinel)

    # find the shallowest product point in each local window
    kernel    = np.ones((OCCLUSION_WIN_PX,)*2, np.float32)
    local_min = cv2.erode(prod_depth.astype(np.float32), kernel)

    keep = depth_m <= (local_min + OCCLUSION_TOLERANCE)
    print(f"  Occlusion cull -> {(~keep).sum():,} px removed")
    cv2.imwrite('debug_occlusion.png', (~keep).astype(np.uint8) * 255)
    return keep


def wall_removal_mask(depth_m):
    d_norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)

    # measure how much depth varies locally — walls are almost perfectly smooth
    r   = WALL_LOCAL_RADIUS * 2 + 1
    k   = np.ones((r, r), np.float32) / (r * r)
    mu  = cv2.filter2D(d_norm, -1, k)
    mu2 = cv2.filter2D(d_norm**2, -1, k)
    local_std = np.sqrt(np.clip(mu2 - mu**2, 0, None))

    # a wall is far away AND has very low depth variance — that combination is pretty unique
    depth_thr = np.percentile(d_norm, WALL_DEPTH_PERCENTILE)
    is_wall   = (d_norm > depth_thr) & (local_std < WALL_FLATNESS_THRESHOLD)

    # close small gaps so the wall mask is one solid region rather than a bunch of holes
    ksize   = 21
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    is_wall = cv2.morphologyEx(
        is_wall.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    print(f"  Wall removal -> {is_wall.sum():,} px removed "
          f"(depth>{depth_thr:.3f}, std<{WALL_FLATNESS_THRESHOLD})")
    print(f"  [tune] Losing objects? raise WALL_DEPTH_PERCENTILE or lower WALL_FLATNESS_THRESHOLD")
    print(f"  [tune] Walls remain?   lower WALL_DEPTH_PERCENTILE or raise WALL_FLATNESS_THRESHOLD")
    cv2.imwrite('debug_wall_mask.png', is_wall.astype(np.uint8)*255)
    return ~is_wall


def save_ply(pts, colors, path):
    # trimesh wants 0-255 RGBA, so convert our 0-1 floats and staple on a full-opacity alpha channel
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    alpha     = np.full((len(pts), 1), 255, dtype=np.uint8)
    rgba      = np.concatenate([colors_u8, alpha], axis=1)

    cloud = trimesh.PointCloud(vertices=pts, colors=rgba)
    cloud.export(path)
    print(f"  PLY saved -> {path}  ({len(pts):,} pts)")
    return cloud


def view_cloud(cloud, pts, colors):
    import pyvista as pv

    # center the cloud so it sits nicely in the viewport
    pts_c = (pts - pts.mean(axis=0)).astype(np.float32)

    cloud_pv = pv.PolyData(pts_c)
    cloud_pv['colors'] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    pl = pv.Plotter(window_size=[1400, 900], title="TEQUILA v5 - Point Cloud")
    pl.background_color = '#14141a'
    pl.add_points(
        cloud_pv,
        scalars='colors',
        rgb=True,
        point_size=3,
        render_points_as_spheres=False
    )
    pl.add_axes()
    pl.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1/5] Device: {device}")

    print("[2/5] Loading DepthAnythingV2...")
    model = DepthAnythingV2(encoder='vits', features=64,
                             out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
    model.to(device).eval()

    print("[3/5] Segmenting + depth inference...")
    raw = cv2.imread(IMG_PATH)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {IMG_PATH}")

    # resize to our working resolution while preserving aspect ratio
    oh, ow = raw.shape[:2]
    ih     = int(oh * INFER_WIDTH / ow)
    img    = cv2.resize(raw, (INFER_WIDTH, ih), interpolation=cv2.INTER_AREA)
    h, w   = img.shape[:2]
    print(f"  {ow}x{oh} -> {w}x{h}")

    product_mask = segment_product(img)

    # replace background with neutral gray before depth inference so the model
    # isn't confused by whatever happens to be behind the product
    masked_input = img.copy()
    masked_input[product_mask == 0] = BG_FILL_COLOR
    cv2.imwrite('debug_masked_input.png', masked_input)

    with torch.no_grad():
        depth_raw = model.infer_image(masked_input)

    # smooth out depth noise while keeping edges sharp
    depth_raw = cv2.bilateralFilter(
        depth_raw.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

    # the model outputs inverse depth, so we flip it and scale to real-world meters
    dmin, dmax = depth_raw.min(), depth_raw.max()
    depth_m    = (1.0 - (depth_raw - dmin) / (dmax - dmin)) * MAX_DEPTH_M

    dvis = ((depth_m / depth_m.max()) * 255).astype(np.uint8)
    cv2.imwrite('debug_depth.png', cv2.applyColorMap(dvis, cv2.COLORMAP_INFERNO))

    print("[4/5] Building point cloud...")

    # back-project every pixel into 3D using a pinhole camera model
    focal  = w / (2.0 * np.tan(np.radians(FOV_H_DEG / 2.0)))
    cx, cy = w / 2.0, h / 2.0
    px, py = np.meshgrid(np.arange(w, dtype=np.float32),
                          np.arange(h, dtype=np.float32))
    x3 = (px - cx) * depth_m / focal
    y3 = (py - cy) * depth_m / focal
    z3 = depth_m

    # stack up all the masks — we only want foreground, non-occluded, non-wall points
    mask = product_mask > 0
    mask = mask & raycast_occlusion_mask(depth_m, product_mask)
    if REMOVE_WALLS:
        mask = mask & wall_removal_mask(depth_m)
    if FLOOR_NAV_MODE:
        hag  = CAMERA_HEIGHT - y3
        mask = mask & (hag > HEIGHT_MIN) & (hag < HEIGHT_MAX)

    pts    = np.stack([x3[mask], y3[mask], z3[mask]], axis=-1)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors = rgb[mask] / 255.0

    print(f"  Raw points: {len(pts):,}")
    if len(pts) == 0:
        print("ERROR: Zero points. Check debug_*.png files.")
        return

    # flip Y and Z so the cloud sits upright with Z going into the scene
    pts[:, 1] *= -1
    pts[:, 2] *= -1

    pts, colors = voxel_downsample(pts, colors, VOXEL_SIZE)
    print(f"  After downsample: {len(pts):,} points")

    print("[5/5] Exporting...")
    cloud = save_ply(pts, colors, OUTPUT_PLY)

    # render a top-down bird's eye view — handy for checking the overall shape at a glance
    x_b, z_b = pts[:, 0], pts[:, 2]
    res  = 0.005
    xi   = ((x_b - x_b.min()) / res).astype(int)
    zi   = ((z_b - z_b.min()) / res).astype(int)
    grid = np.zeros((zi.max()+2, xi.max()+2), dtype=np.uint8)
    grid[np.clip(zi, 0, grid.shape[0]-1),
         np.clip(xi, 0, grid.shape[1]-1)] = 255
    cv2.imwrite(OUTPUT_BEV, cv2.dilate(grid, np.ones((3,3), np.uint8)))
    print(f"  BEV -> {OUTPUT_BEV}")

    view_cloud(cloud, pts, colors)


if __name__ == '__main__':
    main()