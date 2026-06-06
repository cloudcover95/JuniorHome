# JuniorHome

**Current Ecosystem State**

**Added VisionTextEngine for Instagram Story Zoom Tag Inference**

New `VisionTextEngine` in the comparison pipeline specifically handles the use case you described:

- Zoomed video layers in Instagram stories where @account tags are embedded as text.
- Interactive links are dead, but the text is visible in the paused/zoomed frame.
- The engine combines con-layer style features with quant LLM / theoretical math to detect and reason about the embedded account tags.

A test template is included so you can plug in your real theoretical math or BitNet vision models and benchmark how well different engines handle this "dead link but visible text" scenario.

This extends the ecosystem's inference capabilities into practical social media / content analysis while staying sovereign and edge-native.