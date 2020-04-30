from setuptools import setup

setup(name='gb_rf_evolution',
      version='0.1',
      description='genetic algorithm for GB and RF hyperparameter tuning',
      url='https://github.com/EvanBagis/gb_rf_evolution',
      author='Evan Bagis',
      author_email='evanbagis@gmail.com',
      license='MIT',
      packages=['gb_rf_evolution'],
      install_requires=['tqdm', 'numpy', 'logger'],
      zip_safe=False)
