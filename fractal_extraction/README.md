# Model Extraction Attacks without Natuaral Images

This project contains codes for the following paper:

    Kota Yoshida, Takeshi Fujino, "Model Extraction Attacks without Natural Images," ACNS Workshop on Security in Machine Learning and its Applications (SiMLA 2024).

- Affiliations of authors: Ritsumeikan University, Japan  

## Requirements
- Deep learning-enabled PC (Large DRAMs, GPUs)
- Nvidia-Docker

## Author's environment
- Software
    - Ubuntu 20.04 LTS
    - (Nvidia-)Docker 20.10.12
    - Docker-compose 1.29.2
    - Nvidia-driver 510.47.03
    - CUDA 11.6

- Hardware
    - Intel Xeon Gold 6226R *2
    - DDR4 196GB
    - Nvidia A5000 *5 (Partially used)

## Setups

Note: Please modify `fractaldb_docker/docker-compose.yaml` based on your environments.  
(e.g. Service name, Allocated ports, Shared-memory size, Number of GPUs)

1. (Optional) Prepare datasets
    
    - FractalDB public data
        1. Visit Fractal-dataset [project page](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)

        1. Download `FractalDB-1k` and/or `FractalDB-60` from this page

        1. Unzip download file under `dataset/fractal_pub/`

            E.g.:
            - `dataset/fractal_pub/FractalDB-1k/00000/00000_00_count_0_flip0.png, ...`,
            - `dataset/fractal_pub/fractaldb_cat60_ins1000/a1/a1_weight00_patch0_flip0.png, ...`

1. Start docker container
    ```
    cd <this-project-folder>/docker
    docker-compose up
    ```

    Note: This container starts with jupyter server for development, but demo-code is written for command-line execution.

1. Attach to the container 

    ```
    docker ps
    # Check container ID started above.
    docker exec -it <container_ID> /bin/bash
    # DON'T use 'docker attach' command to attach shells because the container is running jupter server in the preimary shell.
    ```

1. Enjoy!

Note:
If you use Microsoft VSCode and Docker extension, you can easy to attach the started docker container with `Attach with Visual Studio Code` menu.
