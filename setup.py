from setuptools import setup, find_packages


setup(name='fsdet',
      version='0.1',
      description='Few-shot object detection and training tools',
      url='https://github.com/wbailer/few-shot-object-detection',
      author='JOANNEUM RESEARCH Forschungsgesellschaft mbH',
      author_email='werner.bailer@joanneum.at',
      license='Apache 2.0',
      packages=find_packages(exclude=('demo','tools','datasets')),
      install_requires=[
          'termcolor',
          'numpy',
          'tqdm',
          'matplotlib',
          'yacs',
          'tabulate',
          'cloudpickle',
          'Pillow',
          'imagesize',
          'tensorboard',
          'opencv-python',
          'lvis'
      ],
      


      
      zip_safe=False)

