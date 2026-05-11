#creating new virtual enviroment 

python3.10 -m venv .venv
source .venv/bin/activate


#trying to install 

pip install torchcrepe

-> Building wheels for collected packages: llvmlite
  Building wheel for llvmlite (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for llvmlite (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [83 lines of output]
      /private/var/folders/jk/n_89wnv11kvdmf6lj1ff7chw0000gn/T/pip-build-env-s22e_l8d/overlay/lib/python3.11/site-packages/setuptools/_vendor/wheel/bdist_wheel.py:4: FutureWarning: The 'wheel' package is no longer the canonical location of the 'bdist_wheel' command, and will be removed in a future release. Please update to setuptools v70.1 or later which contains an integrated version of this command.
        warn(
      running bdist_wheel
      -- The C compiler identification is AppleClang 16.0.0.16000026
      -- The CXX compiler identification is AppleClang 16.0.0.16000026
      -- Detecting C compiler ABI info

->nd
          cmd_obj.run()
        File "<string>", line 170, in run
        File "<string>", line 62, in build_library_files
        File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/subprocess.py", line 571, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['/usr/local/Caskroom/miniforge/base/envs/transcription/bin/python3.11', '/private/var/folders/jk/n_89wnv11kvdmf6lj1ff7chw0000gn/T/pip-install-dgl5i599/llvmlite_9da7c854eaef43aead7522870a37ba12/ffi/build.py']' returned non-zero exit status 1.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for llvmlite
Failed to build llvmlite
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> llvmlite

# concerns: why the fuck is llvmlite not listed as a dependency?

conda install conda-forge::llvmlite

#run again without PIP trying to resolve half the bullshit
pip install --no-deps torchcrepe

#then
python try_crep.py

->     import librosa
ModuleNotFoundError: No module named 'librosa'

#tried pydebs but that requires graphviz which needs a javaruntime. so i am not touching that shit
#try installing librosa

conda install conda-forge::librosa

#about dependecies: if it is listed in setup.py, it might not be there when you pip install. This is fucking annoying because 
# there is no project.toml and setup.py is never downloaded in the package. so it is assumed that torchcrepe already has some of that stuff
# but it seems like it didnt. and couldnt not install itself until i installed llvmlite seperately 
 
#try again
->
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.4.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/Users/erniewang/transcription-sandbox/crepe/try_crep.py", line 1, in <module>
    import torchcrepe
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torchcrepe/__init__.py", line 1, in <module>
    from . import decode
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torchcrepe/decode.py", line 3, in <module>
    import torch
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/__init__.py", line 1477, in <module>
    from .functional import *  # noqa: F403
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/functional.py", line 9, in <module>
    import torch.nn.functional as F
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/nn/__init__.py", line 1, in <module>
    from .modules import *  # noqa: F403
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/nn/modules/__init__.py", line 35, in <module>
    from .transformer import TransformerEncoder, TransformerDecoder, \
  File "/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/nn/modules/transformer.py", line 20, in <module>
    device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
/usr/local/Caskroom/miniforge/base/envs/transcription/lib/python3.11/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
DEBUG: prining out audio tensor([[0., 0., 0.,  ..., 0., 0., 0.]])
zsh: segmentation fault  python try_crep.py

#uninstal and reinstall numby

conda install 'numpy<2' #try again

-> /__init__.py, line 212, in <module>
    import lazy_loader as lazy
ModuleNotFoundError: No module named lazy_loader

#before it was librosa. then typing extensions, then numba. so wiping the slate clean and restarting

conda create -n torchcrepe-env python=3.10 -y
conda activate torchcrepe-env

conda install -c conda-forge \
  "numpy<2" \
  librosa \
  numba \
  llvmlite \
  scipy \
  resampy \
  tqdm \
  pytorch \
  torchaudio \
  -y

pip install --no-deps torchcrepe

python -c "import torchcrepe; print('ok')"

#try again 
DEBUG: prining out audio tensor([[0., 0., 0.,  ..., 0., 0., 0.]])
Traceback (most recent call last):
  File "/Users/erniewang/transcription-sandbox/crepe/try_crep.py", line 26, in <module>
    pitch = torchcrepe.predict(audio,
  File "/usr/local/Caskroom/miniforge/base/envs/torchcrepe-env/lib/python3.10/site-packages/torchcrepe/core.py", line 117, in predict
    for frames in generator:
  File "/usr/local/Caskroom/miniforge/base/envs/torchcrepe-env/lib/python3.10/site-packages/torchcrepe/core.py", line 691, in preprocess
    frames = frames.to(device)
  File "/usr/local/Caskroom/miniforge/base/envs/torchcrepe-env/lib/python3.10/site-packages/torch/cuda/__init__.py", line 417, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled

#change device into cpu instead of CUDA
# try again: 
  File "/usr/local/Caskroom/miniforge/base/envs/torchcrepe-env/lib/python3.10/site-packages/torch/serialization.py", line 710, in default_restore_location
    raise RuntimeError(
RuntimeError: dont know how to restore data location of torch.storage.UntypedStorage (tagged with cpu:0)

#did ms5 and worked -> no result because no printing