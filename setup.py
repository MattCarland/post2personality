from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='post2personality',
      version="0",
      description="MBTI Predicition Model",
      license="MIT",
      author="~",
      author_email="~",
    #   url="https://github.com/MattCarland/post2personality",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
