"""
Run this ONCE locally to generate test_input.json.
It creates a simple test image and encodes it as base64.

Usage:
    python generate_test_input.py
"""

import base64
import json
from PIL import Image, ImageDraw
import io

# Create a simple 512x512 test image (red square on white background)
img = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)
# Draw a red square in the center (simple "object" for testing)
draw.rectangle([100, 100, 400, 400], fill=(200, 60, 60, 255))
# Add a highlight to make it slightly 3D-like
draw.rectangle([100, 100, 250, 250], fill=(230, 90, 90, 255))

# Convert image to base64
buffer = io.BytesIO()
img.save(buffer, format="PNG")
img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# Write test_input.json
test_data = {
    "input": {
        "image": img_base64,
        "seed": 42
    }
}

with open("test_input.json", "w") as f:
    json.dump(test_data, f)

print("test_input.json created!")
print(f"  Image size: 512x512 pixels")
print(f"  Base64 length: {len(img_base64):,} characters")
print()
print("Now run: docker run --rm yourusername/trellis-api:v1.0.0")
