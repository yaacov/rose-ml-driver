# rose-ml-driver

Demo driver for the ROSE project using machine learning.

ROSE project: https://github.com/RedHat-Israel/ROSE

Run the driver:

``` bash
# Get help
podman run --rm --network host -it quay.io/rose/rose-ml-driver:latest --help

# Run the driver on localhost port 8082 (default port in 8081)
podman run --rm --network host -it quay.io/rose/rose-ml-driver:latest --port 8082
```

Run the server:

``` bash
# Start the ROSE game server, and connect to the Go driver
podman run --rm \
  --network host \
  -it quay.io/rose/rose-server:latest \
  --drivers http://127.0.0.1:8082
```

Browse to http://127.0.0.1:8880 to run the game.

Train and run locally:

```bash
# Train a new model
python train.py

# Copy model to checkpoint directory
cp driver.pth ./checkpoints

# Run ML driver (using port 8082, default is port 8081)
python client/main.py -d ./driver.py --port 8082
```
