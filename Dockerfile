FROM nvcr.io/nvidia/pytorch:24.03-py3


# Use the noninteractive frontend for Debian-based packages
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    libgtk2.0-dev \
    pkg-config \
    x11-apps \
    libgl1-mesa-glx \
    python3-tk \
    # add anything extra then do \
    && rm -rf /var/lib/apt/lists/*

# Reset DEBIAN_FRONTEND to its default value
ENV DEBIAN_FRONTEND=
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Assuming your application doesn't need additional Python packages
# beyond what's included in the NVIDIA image, you might skip the pip install step.
# If you do have additional dependencies, you can install them as follows:
COPY requirements.txt /app
##RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall opencv-python
RUN pip install opencv-python==4.5.4.60
RUN pip install ultralytics
RUN pip install transformers
RUN pip install nltk
RUN pip install keyboard
RUN pip install gradio
RUN pip install sentencepiece
RUN pip install langdetect
RUN pip install dill
RUN pip install open-clip-torch
RUN pip install webdataset
