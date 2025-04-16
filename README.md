To run caboose functionalities:
git clone caboose
git clone caboose_index

### Preliminaries
 * Make sure you have Python 3.9 and Rust installed: https://www.rust-lang.org/tools/install
 * Make sure to have https://github.com/schelterlabs/caboose_index checked out as `caboose_index` in the same folder as this project

conda create --name caboose python=3.9
conda activate caboose

cd caboose
 * Setup a virtualenv `python3.9 -m venv venv` and `source venv/bin/activate`
 * Install Cython (needed for similaripy) `pip install Cython==0.29.32`
 * Install the dependencies `pip install -r requirements.txt`
 * Build the project with `maturin develop --release`

conda deactivate