python3 create_adversarial_examples.py --multirun \
    victim=mnist \
    victim.model=small,medium,large \
    attack=mnist,fractaldb_1k \
    attack.model=small,medium,large \
    create_AEs.round="'range(100)'"

python3 create_adversarial_examples.py --multirun \
    victim=fashion_mnist \
    victim.model=small,medium,large \
    attack=fashion_mnist,fractaldb_1k \
    attack.model=small,medium,large \
    create_AEs.round="'range(100)'"
