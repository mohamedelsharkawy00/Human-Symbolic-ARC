#version 450
#extension GL_ARB_shader_draw_parameters : enable

// ----------------------------------------------------------------------------
// CUSTOM ARC PALETTE - "Human Eye Equivalent Brightness"
// ----------------------------------------------------------------------------
// We have manually adjusted these colors to reduce the blinding brightness
// of pure colors like Green/Cyan/Yellow, creating a more pastel/balanced look.
//
// 0: Black       (0.00, 0.00, 0.00)
// 1: Blue        (0.20, 0.40, 0.70) -> Dimmed
// 2: Red         (0.80, 0.30, 0.30) -> Pastel Red
// 3: Green       (0.30, 0.60, 0.30) -> Dimmed from Neon Green
// 4: Yellow      (0.85, 0.75, 0.30) -> Wheat/Gold
// 5: Grey        (0.60, 0.60, 0.60) -> Unchanged Mid-Grey
// 6: Magenta     (0.70, 0.30, 0.60) -> Deep Purple/Magenta
// 7: Orange      (0.85, 0.50, 0.20) -> Burnt Orange
// 8: Cyan        (0.30, 0.65, 0.75) -> Teal
// 9: Maroon      (0.57, 0.07, 0.19) -> Unchanged (already dark)
// 10: White      (1.00, 1.00, 1.00) -> Text/Lines
// 11: Dark Grey  (0.20, 0.20, 0.20)
// 12: Light Grey (0.70, 0.70, 0.70)
// ----------------------------------------------------------------------------
// Palette now supports 0-29
// 0-9: Standard ARC Vivid Colors
// 10-12: Greyscale
// 20-29: Pastel Colors for Tree Nodes
const vec3 palette[30] = vec3[](
    // --- Standard ARC Vivid (0-9) - Exact Values ---
    vec3(0.0),                                      // 0: Black
    vec3(0.118, 0.576, 1.0),                        // 1: Blue
    vec3(0.976, 0.235, 0.192),                      // 2: Red
    vec3(0.310, 0.800, 0.188),                      // 3: Green
    vec3(1.0, 0.863, 0.0),                          // 4: Yellow
    vec3(0.60),                                     // 5: Grey
    vec3(0.898, 0.227, 0.639),                      // 6: Magenta
    vec3(1.0, 0.522, 0.106),                        // 7: Orange
    vec3(0.529, 0.847, 0.945),                      // 8: Cyan
    vec3(0.573, 0.071, 0.192),                      // 9: Maroon

    // --- Greyscale Extras (10-19) ---
    vec3(1.00),                     // 10: White
    vec3(0.20),                     // 11: Dark Grey
    vec3(0.70),                     // 12: Light Grey
    vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), // 13-19 Padding

    // --- Pastel Tree Palette (20-29) ---
    // Tweaked for Separation: Red/Magenta boosted to ~63%, Maroon kept at ~45%
    vec3(0.00, 0.00, 0.00),         // 20: Black
    vec3(0.06, 0.29, 0.50),         // 21: Blue (50%)
    vec3(0.61, 0.15, 0.12),         // 22: Red (Boosted ~63% of orig)
    vec3(0.16, 0.40, 0.09),         // 23: Green (50%)
    vec3(0.50, 0.43, 0.00),         // 24: Yellow (50%)
    vec3(0.30, 0.30, 0.30),         // 25: Grey (50%)
    vec3(0.56, 0.14, 0.40),         // 26: Magenta (Boosted ~63%)
    vec3(0.50, 0.26, 0.05),         // 27: Orange (50%)
    vec3(0.27, 0.42, 0.47),         // 28: Cyan (50%)
    vec3(0.25, 0.03, 0.08)          // 29: Maroon (Darker ~45%)
);

const vec2 quad[6] = vec2[](
    vec2(0.0), vec2(1.0, 0.0), vec2(1.0),
    vec2(0.0), vec2(1.0), vec2(0.0, 1.0)
);

layout(binding = 0, std430) readonly buffer GridData
{
    int cells[];
} grid;

layout(push_constant) uniform PushConstants
{
    int gridW;
    int gridH;
    int screenW;
    int screenH;
    int dataOffset;
    int cellW;
    int startX;
    int startY;
    int stride;
} push;

layout(location = 0) out vec4 fragColor;

void main()
{
    int cellIndex = gl_InstanceIndex;
    int col = cellIndex % push.gridW;
    int row = cellIndex / push.gridW;

    int colorCode = grid.cells[push.dataOffset + ((row * push.stride) + col)];

    // -1 is the transparent sentinel (environment::TRANSPARENT): emit alpha=0
    // so the cell shows through to whatever is behind it.
    if (colorCode == -1) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    } else {
        if (colorCode < 0 || colorCode > 29) {
            colorCode = 0;
        }
        fragColor = vec4(palette[colorCode], 1.0);
    }

    float cellSide = float(push.cellW);
    float px = float(push.startX) + (float(col) * cellSide);
    float py = float(push.startY) + (float(row) * cellSide);

    vec2 vOff = quad[gl_VertexIndex];
    float drawSize = cellSide;

    // GAP LOGIC (Optional: Keep gaps for Grid, Remove for Tree?)
    // If you want gaps in the Tree, keep this. If you want solid blocks, remove it.
    // Given the user liked "bigger backgrounds", solid blocks might be better,
    // but the original shader had gaps. Let's keep a 1px gap for aesthetic grid look
    // unless it's very small.
    // Gap Logic: Apply to large cells (grids) to ensure separation.
    // Avoid applying to small cells (text pixels) to keep font contiguous.
    // Text scale 3 -> cellW=3. We want solid text.
    // Grids usually have cellW >= 10.
    if (push.cellW > 4)
    {
       drawSize = cellSide - 1.0;
    }

    float finalX = px + (vOff.x * drawSize);
    float finalY = py + (vOff.y * drawSize);

    // Convert to NDC (-1 to 1)
    float ndcX = ((finalX / float(push.screenW)) * 2.0) - 1.0;
    float ndcY = ((finalY / float(push.screenH)) * 2.0) - 1.0;

    gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
}
