# Adapted from https://github.com/NVIDIA/waveglow under the BSD 3-Clause License.

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
import tempfile
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
import os
from tqdm import tqdm
from scipy.io.wavfile import write as wavwrite

from dataset import load_CleanNoisyPairDataset
from datasets import load_dataset
from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet
import torchaudio
import soundfile as sf

def load_simple(filename):
    audio, _ = torchaudio.load(filename)
    return audio
    


def denoise(files, ckpt_path, exp_path, batch_size):
    """
    Denoise audio

    Parameters:
    output_directory (str):         save generated speeches to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    subset (str):                   training, testing, validation
    dump (bool):                    whether save enhanced (denoised) audio
    """

    # setup local experiment path
    #exp_path = config["train_config"]["exp_path"]
    #print('exp_path:', exp_path)

    # load data
    loader_config = deepcopy(trainset_config)
    loader_config["crop_length_sec"] = 0



    # predefine model
    net = CleanUNet(**network_config).cuda()
    #print_size(net)

    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    denoised_data = []

    # inference
    for file_path in tqdm(files, disable=(len(files) == 1)):
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_name)
        new_file_name = file_name + "_denoised.wav"
        noisy_audio = load_simple(file_path).cuda()
        LENGTH = len(noisy_audio[0].squeeze())
        noisy_audio = torch.chunk(noisy_audio, LENGTH // batch_size + 1, dim=1)
        all_audio = []

        for batch in tqdm(noisy_audio, disable=(len(noisy_audio) == 1)):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    generated_audio = sampling(net, batch)
                    generated_audio = generated_audio.cpu().numpy().squeeze()
                    all_audio.append(generated_audio)
        
        all_audio = np.concatenate(all_audio, axis=0)
        denoised_data.append(all_audio.squeeze())
        save_file = os.path.join(file_dir, new_file_name)
        #print("saved to:", save_file)
        wavwrite(save_file, 
                16000,
                all_audio.squeeze())
    return denoised_data




# Modify this part of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_path', '--ckpt_path',
                        help='Path to the checkpoint you want to use')     
    parser.add_argument('-b', '--batch_size', type=int, help='chunk your input audio vector into chunks of batch_size. not exact.', default=100_000)
     
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Path to the JSON file containing the dataset')

    args = parser.parse_args()
    


    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    gen_config              = config["gen_config"]
    global network_config
    network_config          = config["network_config"]      # to define wavenet
    global train_config
    train_config            = config["train_config"]        # train config
    global trainset_config
    trainset_config         = config["trainset_config"]     # to read trainset configurations
    bs = args.batch_size
    exp_path = config["train_config"]["exp_path"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Save the audio files to temporary locations and pass their paths to the denoise function
    temp_files = []
    denoised_audios = []
    with open(args.dataset) as f:
      dataset = json.load(f)
    # Iterate through the list of dictionaries and process the dataset
    for example in dataset:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            audio_data = np.array(example['audio']['array']).reshape(-1, 1)
            sf.write(temp_file.name, audio_data, samplerate=16000)
            denoised_audio = denoise([temp_file.name], args.ckpt_path, exp_path, batch_size=args.batch_size)
            denoised_audios.append(denoised_audio[0])
            temp_files.append(temp_file.name)
    # Create a new dataset with denoised audio data and sentences
    denoised_dataset = Dataset.from_dict({
    'audio': denoised_audios,
    'sentence': [example['sentence'] for example in dataset]})

    print(denoised_dataset)
    denoised_dataset_list = [dict(example) for example in denoised_dataset]
    with open("denoised_dataset.json", "w") as f:
      json.dump(denoised_dataset_list, f)

    
