# cuda toolkit Archive -> https://developer.nvidia.com/cuda-toolkit-archive

# python 3.11 install
echo "Installing python ...."
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 -y
sudo apt install python3-pip -y
sudo apt install python3-venv -y
sudo apt install wget -y
sudo apt install dpkg -y
sudo apt install git -y

# update pip
echo "Updating pip ...."
python3 -m pip install --upgrade pip

# miniconda install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# conda env create
echo "Creating conda environment ...."
conda create -n myenv python=3.11 -y
conda activate myenv && echo "Conda environment activated ...."

# cuda install
echo "Installing cuda ...."
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# cuda path
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# install requirements
pip3 uninstall -y torch packaging ninja flash-attn
echo "Installing requirements ...."
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install packaging
pip3 install ninja
pip3 install flash-attn --no-build-isolation

# xformers install (stable diffusion)
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

# t5 (text to speech)
pip3 install --upgrade sentencepiece datasets[audio]

# speech to text requirements
pip3 install torchaudio
pip3 install ffmpeg

pip3 install -r requirements.txt

# unsloth install
echo "Installing unsloth ...."
pip3 install "unsloth[cu118_ampere_torch220] @ git+https://github.com/unslothai/unsloth.git"


