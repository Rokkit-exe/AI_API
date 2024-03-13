#!/bin/bash
miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh"
cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
cuda_deb_url="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb"
cuda_deb="cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb"
torch_url="https://download.pytorch.org/whl/cu118"

# -------------------------------------------------------- Functions --------------------------------------------------------/
function print_wait {
    printf '\033[32m\n\n %s .... \n\n\033[0m' "$1"
    sleep 2
}

function install_apt_package {
    package_name=$1
    print_wait "Installing $package_name"
    apt install --upgrade $package_name -y
}

function update_system {
    print_wait "Updating system"
    apt update -y && apt full-upgrade -y && apt autoremove -y && apt autoclean -y
}

function install_miniconda {
    print_wait "Installing miniconda"
    mkdir -p ~/miniconda3
    print_wait "Downloading miniconda"
    wget $miniconda_url -O ~/miniconda3/miniconda.sh
    print_wait "Installing miniconda"
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    print_wait "list ~/miniconda3"
    ls ~/miniconda3
    print_wait "Removing miniconda.sh"
    rm -rf ~/miniconda3/miniconda.sh
    print_wait "Initializing conda"
    ~/miniconda3/bin/conda init bash
    print_wait "Modifying PATH"
    export PATH="~/miniconda3/bin:$PATH"
    print_wait "Sourcing bashrc"
    source ~/.bashrc
}

function create_conda_env {
    print_wait "Creating conda environment name:myenv, python version:3.10"
    conda create -n myenv python=3.10 -y
    print_wait "Activating conda environment"
    conda activate myenv && echo "Conda environment activated ...."
}

function install_cuda {
    print_wait "Installing cuda for wsl Ubuntu"
    wget $cuda_repo_url
    print_wait "Moving cuda-wsl-ubuntu.pin to /etc/apt/preferences.d/cuda-repository-pin-600"
    mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    print_wait "Downloading $cuda_deb"
    wget $cuda_deb_url
    print_wait "Installing $cuda_deb"
    dpkg -i $cuda_deb
    print_wait "Copying cuda-*-keyring.gpg to /usr/share/keyrings/"
    cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    print_wait "Updating apt"
    apt update -y
    print_wait "Installing cuda"
    apt -y install cuda
}

function set_cuda_path {
    print_wait "Setting cuda path"
    echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    print_wait "Sourcing bashrc"
    source ~/.bashrc
}

function install_pip_package {
    package_name=$1
    index_url=$2
    if [ -z "$index_url" ]
    then
        print_wait "Installing $package_name"
        pip3 install --upgrade $package_name
    else
        print_wait "Installing $package_name from $index_url"
        pip3 install --upgrade $package_name --index-url $index_url
    fi
}

# -------------------------------------------------------- Execution --------------------------------------------------------
update_system

install_apt_package "python3"
install_apt_package "python3-pip"
install_apt_package "python3-venv"

install_apt_package "wget"
install_apt_package "dpkg"
install_apt_package "git"
install_apt_package "ffmpeg" 


print_wait "Updating pip"
python3 -m pip install --upgrade pip


install_miniconda
create_conda_env


install_cuda
set_cuda_path

install_pip_package "torch" "https://download.pytorch.org/whl/cu118"
install_pip_package "packaging"
install_pip_package "ninja"

print_wait "Installing flash-attn"
pip3 install flash-attn --no-build-isolation

install_pip_package "xformers" "https://download.pytorch.org/whl/cu118"
install_pip_package "sentencepiece"
install_pip_package "datasets[audio]"
install_pip_package "torchaudio"

print_wait "Installing requirements"
pip3 install -r requirements.txt