import os
import sys
from PIL import Image, ImageDraw
import svgelements


def render_svg_to_image(svg_path, size=(512, 512), padding=0.0):
    # Parse SVG
    svg = svgelements.SVG.parse(svg_path)
    
    # Create target image
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Get total bounds of all elements
    try:
        bbox = svg.bbox()
        if bbox is None:
            print(f"[!] Could not calculate bounding box for {svg_path}")
            return None
    except Exception as e:
        print(f"[!] Error calculating bbox: {e}")
        return None
        
    x1, y1, x2, y2 = bbox
    svg_w = x2 - x1
    svg_h = y2 - y1
    
    # Target dimensions based on padding
    # If padding=0, target width is the full image width
    target_w = size[0] * (1.0 - 2 * padding)
    target_h = size[1] * (1.0 - 2 * padding)
    
    # Scale to fit width (since it's a wide 5x2 strip)
    scale = target_w / svg_w
    
    # Offsets to center vertically
    off_x = padding * size[0] - x1 * scale
    off_y = (size[1] - svg_h * scale) / 2 - y1 * scale
    
    # Render elements
    for element in svg.elements():
        if isinstance(element, (svgelements.Path, svgelements.Rect, svgelements.Polygon)):
            if element.fill is not None:
                color = element.fill.hexrgb
                points = []
                # Approximate path with points for drawing
                if isinstance(element, svgelements.Path):
                    for segment in element:
                        if isinstance(segment, (svgelements.Move, svgelements.Line)):
                            points.append((segment.end[0] * scale + off_x, segment.end[1] * scale + off_y))
                elif isinstance(element, (svgelements.Rect, svgelements.Polygon)):
                    # For Rect/Polygon, we can get points or build them
                    # Simplest is to treat everything as Path if possible
                    pass
                
                if points:
                    draw.polygon(points, fill=color)

    return img


def generate_resources():
    assets_dir = "assets"
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
    icon_svg = os.path.join(assets_dir, "arc_logo.svg")
    icon_ico = os.path.join(build_dir, "icon.ico")
    app_rc = os.path.join(build_dir, "app.rc")

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # 1. Generate ICO from SVG
    if os.path.exists(icon_svg):
        print(f"[*] Rendering {icon_svg} (zero padding) to {icon_ico}...")
        master_img = render_svg_to_image(icon_svg, (512, 512), padding=0.0)
        if master_img:
            # Standard sizes for Windows ICO
            icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            master_img.save(icon_ico, sizes=icon_sizes)
        else:
            print("[!] Failed to render SVG.")
            return
    else:
        print(f"[!] {icon_svg} not found, skipping icon generation.")
        return

    # 2. Create .rc file
    print(f"[*] Creating {app_rc}...")
    with open(app_rc, "w") as f:
        f.write(f'IDI_ICON1 ICON "{os.path.abspath(icon_ico).replace("\\\\", "/").replace("\\", "/")}"\n')


if __name__ == "__main__":
    generate_resources()
