## Use Docker
```bash
docker build -t {image_name:tag} ./
docker run -it --rm --name torch_container --gpus all -v $PWD:/workspace -w /workspace -p {port}:{port} {image_name:tag} /bin/bash
```