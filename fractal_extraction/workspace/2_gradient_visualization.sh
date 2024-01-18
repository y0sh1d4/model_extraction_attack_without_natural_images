# Apply Grad-CAM
python3 gradient_visualization.py --run \
    victim=fashion_mnist \
    victim.model=large \
    attack=fashion_mnist \
    attack.model=large \
    gv.round=100
    