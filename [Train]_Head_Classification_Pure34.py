{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!gdown 1G2DkYk-vpbTOZJhpgOm9Sj1peYnLqvie"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9v02Plo_zxyK",
        "outputId": "cd085971-f49e-43ac-ad08-6d17e90c8e90"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading...\n",
            "From (original): https://drive.google.com/uc?id=1G2DkYk-vpbTOZJhpgOm9Sj1peYnLqvie\n",
            "From (redirected): https://drive.google.com/uc?id=1G2DkYk-vpbTOZJhpgOm9Sj1peYnLqvie&confirm=t&uuid=94fc7604-1c80-49d5-8fec-88bacdb205af\n",
            "To: /content/cropped_dataset_4k_face.zip\n",
            "100% 44.3M/44.3M [00:01<00:00, 42.6MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch torchvision albumentations pandas"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "T9x10DQE0D8m",
        "outputId": "1ee46e5a-883c-46cd-ee15-a49c07a47ed3"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.21.0+cu124)\n",
            "Requirement already satisfied: albumentations in /usr/local/lib/python3.11/dist-packages (2.0.8)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.14.0)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.5)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)\n",
            "Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch)\n",
            "  Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cublas-cu12==12.4.5.8 (from torch)\n",
            "  Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cufft-cu12==11.2.1.3 (from torch)\n",
            "  Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-curand-cu12==10.3.5.147 (from torch)\n",
            "  Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch)\n",
            "  Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch)\n",
            "  Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch) (0.6.2)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch) (2.21.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)\n",
            "Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch)\n",
            "  Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)\n",
            "Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.2.0)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (2.0.2)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.2.1)\n",
            "Requirement already satisfied: scipy>=1.10.0 in /usr/local/lib/python3.11/dist-packages (from albumentations) (1.15.3)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from albumentations) (6.0.2)\n",
            "Requirement already satisfied: pydantic>=2.9.2 in /usr/local/lib/python3.11/dist-packages (from albumentations) (2.11.7)\n",
            "Requirement already satisfied: albucore==0.0.24 in /usr/local/lib/python3.11/dist-packages (from albumentations) (0.0.24)\n",
            "Requirement already satisfied: opencv-python-headless>=4.9.0.80 in /usr/local/lib/python3.11/dist-packages (from albumentations) (4.11.0.86)\n",
            "Requirement already satisfied: stringzilla>=3.10.4 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.24->albumentations) (3.12.5)\n",
            "Requirement already satisfied: simsimd>=5.9.2 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.24->albumentations) (6.4.9)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2025.2)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.33.2 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (2.33.2)\n",
            "Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (0.4.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)\n",
            "Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m363.4/363.4 MB\u001b[0m \u001b[31m3.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.8/13.8 MB\u001b[0m \u001b[31m121.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m24.6/24.6 MB\u001b[0m \u001b[31m93.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m883.7/883.7 kB\u001b[0m \u001b[31m59.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m664.8/664.8 MB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m211.5/211.5 MB\u001b[0m \u001b[31m5.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m13.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m127.9/127.9 MB\u001b[0m \u001b[31m7.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.5/207.5 MB\u001b[0m \u001b[31m6.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.1/21.1 MB\u001b[0m \u001b[31m93.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12\n",
            "  Attempting uninstall: nvidia-nvjitlink-cu12\n",
            "    Found existing installation: nvidia-nvjitlink-cu12 12.5.82\n",
            "    Uninstalling nvidia-nvjitlink-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-nvjitlink-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-curand-cu12\n",
            "    Found existing installation: nvidia-curand-cu12 10.3.6.82\n",
            "    Uninstalling nvidia-curand-cu12-10.3.6.82:\n",
            "      Successfully uninstalled nvidia-curand-cu12-10.3.6.82\n",
            "  Attempting uninstall: nvidia-cufft-cu12\n",
            "    Found existing installation: nvidia-cufft-cu12 11.2.3.61\n",
            "    Uninstalling nvidia-cufft-cu12-11.2.3.61:\n",
            "      Successfully uninstalled nvidia-cufft-cu12-11.2.3.61\n",
            "  Attempting uninstall: nvidia-cuda-runtime-cu12\n",
            "    Found existing installation: nvidia-cuda-runtime-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-runtime-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-runtime-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cuda-nvrtc-cu12\n",
            "    Found existing installation: nvidia-cuda-nvrtc-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-nvrtc-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cuda-cupti-cu12\n",
            "    Found existing installation: nvidia-cuda-cupti-cu12 12.5.82\n",
            "    Uninstalling nvidia-cuda-cupti-cu12-12.5.82:\n",
            "      Successfully uninstalled nvidia-cuda-cupti-cu12-12.5.82\n",
            "  Attempting uninstall: nvidia-cublas-cu12\n",
            "    Found existing installation: nvidia-cublas-cu12 12.5.3.2\n",
            "    Uninstalling nvidia-cublas-cu12-12.5.3.2:\n",
            "      Successfully uninstalled nvidia-cublas-cu12-12.5.3.2\n",
            "  Attempting uninstall: nvidia-cusparse-cu12\n",
            "    Found existing installation: nvidia-cusparse-cu12 12.5.1.3\n",
            "    Uninstalling nvidia-cusparse-cu12-12.5.1.3:\n",
            "      Successfully uninstalled nvidia-cusparse-cu12-12.5.1.3\n",
            "  Attempting uninstall: nvidia-cudnn-cu12\n",
            "    Found existing installation: nvidia-cudnn-cu12 9.3.0.75\n",
            "    Uninstalling nvidia-cudnn-cu12-9.3.0.75:\n",
            "      Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75\n",
            "  Attempting uninstall: nvidia-cusolver-cu12\n",
            "    Found existing installation: nvidia-cusolver-cu12 11.6.3.83\n",
            "    Uninstalling nvidia-cusolver-cu12-11.6.3.83:\n",
            "      Successfully uninstalled nvidia-cusolver-cu12-11.6.3.83\n",
            "Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VziHvAU-zjXA",
        "outputId": "52346de0-839b-4762-d9f1-bc27b654ce95"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "File sau khi giải nén:\n",
            "['Dog Emotion']\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import zipfile\n",
        "import os\n",
        "\n",
        "zip_path = \"/content/cropped_dataset_4k_face.zip\"\n",
        "extract_path = \"/content/data\"  # nơi chứa dữ liệu sau giải nén\n",
        "\n",
        "with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
        "    zip_ref.extractall(extract_path)\n",
        "\n",
        "# Kiểm tra kết quả\n",
        "print(\"File sau khi giải nén:\")\n",
        "print(os.listdir(extract_path))\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from torchvision import transforms\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import pandas as pd\n",
        "import os\n",
        "from PIL import Image\n",
        "\n",
        "class MyDataset(Dataset):\n",
        "    def __init__(self, root, labels_csv, transform=None):\n",
        "        self.root = root\n",
        "        df = pd.read_csv(labels_csv)\n",
        "        self.items = df[['filename', 'label']].values\n",
        "\n",
        "        # Tạo ánh xạ label → chỉ số\n",
        "        unique_labels = sorted(df['label'].unique())\n",
        "        self.label2index = {name: i for i, name in enumerate(unique_labels)}\n",
        "\n",
        "        self.transform = transform\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.items)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        fn, label_str = self.items[idx]\n",
        "        label_idx = self.label2index[label_str]\n",
        "        img_path = os.path.join(self.root, label_str, fn)\n",
        "        img = Image.open(img_path).convert('RGB')\n",
        "        if self.transform:\n",
        "            img = self.transform(img)\n",
        "        return img, label_idx\n",
        "\n",
        "\n",
        "\n",
        "transform = transforms.Compose([\n",
        "    transforms.RandomResizedCrop(512),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485,0.456,0.406],\n",
        "                         std=[0.229,0.224,0.225]),\n",
        "])\n",
        "dataset = MyDataset('/content/data/Dog Emotion', '/content/data/Dog Emotion/labels.csv', transform)\n",
        "loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)\n"
      ],
      "metadata": {
        "id": "yb2K_XcOzysp"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import importlib\n",
        "import pure34\n",
        "import torch.nn as nn\n",
        "from pure34 import PURe34\n",
        "importlib.reload(pure34)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "odfLzSl_1NrA",
        "outputId": "dadb1ff8-911f-4820-bc76-e48d0da41997"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<module 'pure34' from '/content/pure34.py'>"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "model = PURe34(num_classes=4).to(device)\n",
        "opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)\n",
        "sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)\n",
        "crit = nn.CrossEntropyLoss()\n",
        "\n",
        "for epoch in range(20):\n",
        "    model.train()\n",
        "    total, correct = 0,0\n",
        "    try:\n",
        "      for imgs, labels in loader:\n",
        "          imgs, labels = imgs.to(device), labels.to(device)\n",
        "          opt.zero_grad()\n",
        "          logits = model(imgs)\n",
        "          loss = crit(logits, labels)\n",
        "          loss.backward()\n",
        "          opt.step()\n",
        "\n",
        "          pred = logits.argmax(dim=1)\n",
        "          total += labels.size(0)\n",
        "          correct += (pred==labels).sum().item()\n",
        "    except FileNotFoundError:\n",
        "        print('error')\n",
        "        continue\n",
        "    sched.step()\n",
        "    print(f'Epoch {epoch+1}  Acc={correct/total:.4f}  LR={sched.get_last_lr()[0]:.4f}')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "hCwm55d00fba",
        "outputId": "365e9392-84cb-49d1-a4b4-19a093852bd3"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "error\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            "    self._shutdown_workers()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "    if w.is_alive():\n",
            "  Exception ignored in:  <function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>\n",
            " Traceback (most recent call last):\n",
            "   File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            "      self._shutdown_workers()^\n",
            "^  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "^    ^if w.is_alive():^\n",
            "^ ^ ^ ^ ^^  ^ \n",
            "^  File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "    ^assert self._parent_pid == os.getpid(), 'can only test a child process'^\n",
            "     ^ ^^ ^^^^ ^^\n",
            "  File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "    assert self._parent_pid == os.getpid(), 'can only test a child process'\n",
            "             ^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "^AssertionError: can only test a child process\n",
            "^^^^^^^^^^^\n",
            "AssertionError: can only test a child process\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "error\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>\n",
            "Traceback (most recent call last):\n",
            "Exception ignored in:   File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            "<function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>\n",
            "    self._shutdown_workers()Traceback (most recent call last):\n",
            "\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "        if w.is_alive():\n",
            "  self._shutdown_workers()\n",
            "   File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "        if w.is_alive():^\n",
            "^ ^ ^^  ^^ ^  ^^^^^^^^^^^\n",
            "  File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "^    assert self._parent_pid == os.getpid(), 'can only test a child process'\n",
            "^ ^ ^^\n",
            "  File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "     assert self._parent_pid == os.getpid(), 'can only test a child process' \n",
            "           ^^ ^ ^ ^ ^^ ^ ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "AssertionError^: ^^can only test a child process^\n",
            "^^^Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>^\n",
            "\n",
            "Traceback (most recent call last):\n",
            "AssertionError  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            ":     can only test a child processself._shutdown_workers()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "    \n",
            "if w.is_alive():\n",
            "Exception ignored in:   <function _MultiProcessingDataLoaderIter.__del__ at 0x7ab8af76e480>    \n",
            " Traceback (most recent call last):\n",
            "^  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1618, in __del__\n",
            "    ^^self._shutdown_workers()^\n",
            "^  File \"/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py\", line 1601, in _shutdown_workers\n",
            "    ^^if w.is_alive():^\n",
            "^ ^ ^^\n",
            "    File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "      assert self._parent_pid == os.getpid(), 'can only test a child process' \n",
            "^^ ^   ^  ^^  ^ ^ ^ ^^^^^^^\n",
            "^^^  File \"/usr/lib/python3.11/multiprocessing/process.py\", line 160, in is_alive\n",
            "    ^assert self._parent_pid == os.getpid(), 'can only test a child process'^^^\n",
            "^^   ^^ ^^ ^ ^^ ^ ^^ ^  ^^^^^^^^^^^^^^^^^\n",
            "^AssertionError^^^: ^can only test a child process^\n",
            "^^^^^^^^^^^^^^^\n",
            "AssertionError: can only test a child process\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "RuntimeError",
          "evalue": "expand(torch.cuda.FloatTensor{[8, 1, 64, 128, 128]}, size=[-1, 64, -1, -1]): the number of sizes provided (4) must be greater or equal to the number of dimensions in the tensor (5)",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-34-3637680807.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     12\u001b[0m           \u001b[0mimgs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mimgs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mto\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m           \u001b[0mopt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzero_grad\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 14\u001b[0;31m           \u001b[0mlogits\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimgs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     15\u001b[0m           \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcrit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlogits\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m           \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1737\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_compiled_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[misc]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1738\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1739\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1740\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1741\u001b[0m     \u001b[0;31m# torchrec tests the code consistency with the following code\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1748\u001b[0m                 \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_pre_hooks\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_hooks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1749\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[0;32m-> 1750\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1751\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1752\u001b[0m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/content/pure34.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m     60\u001b[0m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     61\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mlayer\u001b[0m \u001b[0;32min\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlayer1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlayer2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlayer3\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlayer4\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 62\u001b[0;31m             \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlayer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     63\u001b[0m         \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mavgpool\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mview\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     64\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1737\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_compiled_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[misc]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1738\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1739\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1740\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1741\u001b[0m     \u001b[0;31m# torchrec tests the code consistency with the following code\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1748\u001b[0m                 \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_pre_hooks\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_hooks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1749\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[0;32m-> 1750\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1751\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1752\u001b[0m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, input)\u001b[0m\n\u001b[1;32m    248\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    249\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mmodule\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 250\u001b[0;31m             \u001b[0minput\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodule\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    251\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    252\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1737\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_compiled_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[misc]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1738\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1739\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1740\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1741\u001b[0m     \u001b[0;31m# torchrec tests the code consistency with the following code\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1748\u001b[0m                 \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_pre_hooks\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_hooks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1749\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[0;32m-> 1750\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1751\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1752\u001b[0m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/content/pure34.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m     36\u001b[0m         \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbn1\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconv1\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     37\u001b[0m         \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mF\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrelu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 38\u001b[0;31m         \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbn2\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprod\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     39\u001b[0m         \u001b[0mres\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdownsample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     40\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mF\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrelu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mres\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_wrapped_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1737\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_compiled_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# type: ignore[misc]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1738\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1739\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call_impl\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1740\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1741\u001b[0m     \u001b[0;31m# torchrec tests the code consistency with the following code\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m_call_impl\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1748\u001b[0m                 \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_pre_hooks\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0m_global_backward_hooks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1749\u001b[0m                 or _global_forward_hooks or _global_forward_pre_hooks):\n\u001b[0;32m-> 1750\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mforward_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1751\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1752\u001b[0m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/content/pure34.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m         \u001b[0;31m# nhân tích chéo: feature-wise product\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 21\u001b[0;31m         return F.conv2d(torch.exp(torch.log(x+1e-6).unsqueeze(1).expand(-1, x.size(1), -1,-1)), \n\u001b[0m\u001b[1;32m     22\u001b[0m                         self.weight, bias=self.bias, padding=1)\n\u001b[1;32m     23\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mRuntimeError\u001b[0m: expand(torch.cuda.FloatTensor{[8, 1, 64, 128, 128]}, size=[-1, 64, -1, -1]): the number of sizes provided (4) must be greater or equal to the number of dimensions in the tensor (5)"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "hLiuNrbf1K5A"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}