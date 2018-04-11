import multiprocessing
import json
import paramiko


manager = multiprocessing.Manager()

port = manager.Value('i', 0)
host_ip = manager.Value('s', "")

image_dir = manager.Value("s", "/data/deeplearning/dataset/auto_training/images")
package_dir = manager.Value("s", "/data/deeplearning/dataset/auto_training/packages")
check_dir = manager.Value("s", "/data/deeplearning/dataset/auto_training/released")
release_dir = manager.Value("s", "/data/deeplearning/dataset/auto_training/lane")
krs_url = manager.Value("s", "http://192.168.5.31:23100/krs")
release_url = manager.Value("s", "http://192.168.5.31:23300/kts")

version = manager.Value("s", "1.0.0")
model_host = manager.Value("s", "192.168.5.40")
model_user = manager.Value("s", "kddev")
model_port = manager.Value("i", 22)
model_passwd = manager.Value("s", "kd-123")


def init_params(config_path):
    try:
        with open(config_path, "r") as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            key = str(key).strip()
            value = str(value).strip()
            if key == "port":
                port.value = int(value)
            elif key == "hostIp":
                host_ip.value = value
            elif key == "releaseDir":
                release_dir.value = value
            elif key == "imageDir":
                image_dir.value = value
            elif key == "packageDir":
                package_dir.value = value
            elif key == "checkDir":
                check_dir.value = value
            elif key == "krsUrl":
                krs_url.value = value
            elif key == "releaseUrl":
                release_url.value = value
            elif key == "modelHost":
                model_host.value = value
            elif key == "modelPort":
                model_port.value = int(value)
            elif key == "modelUser":
                model_user.value = value
            elif key == "modelPassword":
                model_passwd.value = value
    except Exception as e:
        print (repr(e))
        return False
    return True


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
