from setuptools import setup

setup(name='d2gnn',
      version='0.1',
      description='GNN for defect formation and migration energies (based originally on CGCNN architecture with various updates)',
      author='Matthew Witman',
      author_email='mwitman1@sandia.gov',
      license='MIT',
      packages=['d2gnn'],
      zip_safe=False,
      python_requires='>3.6',
      install_requires=[
        'pymatgen',
        'ase',
        'torch',
        'numpy',
        'scikit-learn',
      ],
      entry_points={
          'console_scripts': ['d2gnn-train=d2gnn.command_line_train:main',
                              'd2gnn-predict=d2gnn.command_line_predict:main'],
      },
    )

