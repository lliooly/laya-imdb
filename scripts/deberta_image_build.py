"""Importable, dependency-free image layer function; no training code imports."""


def install_tokenizer_dependencies():
    import os
    import resource
    import subprocess
    resource.setrlimit(resource.RLIMIT_CPU,(150,150))
    os.environ.update(UV_CONCURRENT_DOWNLOADS='2',UV_CONCURRENT_INSTALLS='1',UV_CONCURRENT_BUILDS='1')
    subprocess.run(['uv','pip','install','--python','/.uv/.venv/bin/python',
                    '--no-deps','sentencepiece==0.2.2','protobuf==6.33.6'],check=True,timeout=120)
