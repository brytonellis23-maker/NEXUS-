NEXUS CAM — VISION-LANGUAGE BUILD

This build adds a vision-language ranking layer. The PC browser first proposes
candidate regions with a general detector, then a CLIP vision-language model
ranks those regions against the natural-language target description.

Examples:
- "red cup"
- "small blue bottle"
- "black backpack"
- "a person wearing a red shirt"

The browser may need to download the vision-language model the first time.
If it cannot load, the app falls back to the general detector.

This is not perfect open-world recognition. Fine-grained recognition of a
specific unusual object may still require a custom trained model.

Automatic pan/tilt commands are discrete correction requests and still require
final Nano ESP32 motor-control integration and low-speed testing with the
physical emergency stop available.
