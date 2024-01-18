# Model extraction attack
# Attack against MNIST-trained victim moddel
python3 model_extraction_attack.py --multirun \
    victim=mnist,fashion_mnist \
    victim.model=medium \
    attack=fractaldb28_1k \
    attack.model=small,medium,large \
    attack.Early_stopping_patience=10 \
    attack.strategy=random_query

# Attack against Fashion-MNIST-trained victim moddel
# python3 model_extraction_attack.py --multirun \
#     victim=fashion_mnist \
#     victim.model=small,medium,large \
#     attack=fashion_mnist,fractaldb_1k \
#     attack.model=small,medium,large \
#     attack.Early_stopping_patience=10 \
#     attack.strategy=random_query