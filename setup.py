from setuptools import setup, find_packages
import sys, os.path


from version import VERSION

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(name='rlconcepts',
      version=VERSION,
      description='RL-concepts: A package exploring key reinforcement learning concepts.',
      author='Guillaume de Chambrier',
      author_email='chambrierg@gmail.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('rl')],
      zip_safe=False,
      install_requires=[
          'gym','numpy>=1.10.4'
      ]
)
