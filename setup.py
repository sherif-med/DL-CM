from setuptools import setup, find_packages

setup(
    name='DL-CM',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/sherif-med/DL-CM',
    license='MIT',
    author='Mohamed Cherif',
    author_email='cherif.abedrazek@gmail.com',
    description='DL-CM is a Python library designed to streamline common tasks in deep learning and foster innovation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "yamale"
    ],
    extras_require={
        'dev': [
            # 'pytest>=5.2',
            # 'black>=19.10b0',
        ]
    },
    include_package_data=True,
    classifiers=[
        'Development Status ::  3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python ::  3.8',
        'Programming Language :: Python ::  3.9',
    ],
    entry_points={
        'console_scripts': [
            # 'your-script = your_package.your_module:main_function',
        ],
    },
)