# Synthetic vs Natural Materials for Façades (CAP6415 F25)

We evaluate the sim-to-real gap for façade material classification across five classes: brick, glass, concrete, metal panel, vegetation. We train the same CNN under three regimes—NAT-only, SYN-only, MIX (50/50)—and report accuracy and F1 on a held-out natural test set. We use Grad-CAM to verify that the model attends to architectural cues (joints, mullions, texture grains), and deploy the best model to TouchDesigner via OSC for real-time visualization/control.

Quickstart:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make td-demo
```
