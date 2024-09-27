from setuptools import setup, find_packages

setup(
    name='AutoCP',
    version='1.0.0',
    author='Felix Tempel',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy==1.23.5',
        'scipy==1.13.1',
        'torch==2.3.1',
        'tqdm==4.66.4',
        'thop==0.1.1.post2209072238',
        'matplotlib==3.9.0',
        'omegaconf==2.2.3',
        'pynvml==11.4.1',
        'setuptools==69.2.0',
        'tensorboard',
    ],
)